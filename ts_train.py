""" 
refer:
- https://github.com/albertcity/OCARL
- https://github.com/pioneer-innovation/Real-3D-Embodied-Dataset

"""
import math
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 
import warnings
warnings.filterwarnings("ignore")

import time
import pprint
import shutil
import random

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import tianshou as ts
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic
from tianshou.trainer import OnpolicyTrainer

import torch.distributed as dist_module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

print(ts.__version__)
import model
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from mycollector import PackCollector
 

def make_envs(args):

    train_envs = ts.env.SubprocVectorEnv(
        [lambda: gym.make(args.env.id, 
                          container_size=args.env.container_size,
                          enable_rotation=args.env.rot,
                          data_type=args.env.box_type,
                          item_set=args.env.box_size_set, 
                          reward_type=args.train.reward_type,
                          action_scheme=args.env.scheme,
                          k_placement=args.env.k_placement) 
                          for _ in range(args.train.num_processes)]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [lambda: gym.make(args.env.id, 
                          container_size=args.env.container_size,
                          enable_rotation=args.env.rot,
                          data_type=args.env.box_type,
                          item_set=args.env.box_size_set, 
                          reward_type=args.train.reward_type,
                          action_scheme=args.env.scheme,
                          k_placement=args.env.k_placement) 
                          for _ in range(1)]
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    return train_envs, test_envs


def build_net(args, device):
    feature_net = model.ShareNet(
        k_placement=args.env.k_placement, 
        box_max_size=args.env.box_big, 
        container_size=args.env.container_size, 
        embed_size=args.model.embed_dim, 
        num_layers=args.model.num_layers,
        forward_expansion=args.model.forward_expansion,
        heads=args.model.heads,
        dropout=args.model.dropout,
        device=device,
        place_gen=args.env.scheme
    )

    actor = model.ActorHead(
        preprocess_net=feature_net, 
        embed_size=args.model.embed_dim, 
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    critic = model.CriticHead(
        preprocess_net=feature_net, 
        k_placement=args.env.k_placement,
        embed_size=args.model.embed_dim,
        padding_mask=args.model.padding_mask,
        device=device, 
    ).to(device)

    return actor, critic


def train(args):

    date = time.strftime(r'%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    time_str = args.env.id + "_" + \
        str(args.env.container_size[0]) + "-" + str(args.env.container_size[1]) + "-" + str(args.env.container_size[2]) + "_" + \
        args.env.scheme + "_" + str(args.env.k_placement) + "_" +\
        args.env.box_type + "_" + \
        args.train.algo  + '_' \
        'seed' + str(args.seed) + "_" + \
        args.opt.optimizer + "_" \
        + date

    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpus > 1:
    # 初始化 DDP
        dist_module.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        #单卡模式
        local_rank = 0
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # environments 
    # —— 自动根据可见 GPU 数量缩放超参数（无需改 config.yaml） —— #
    # 如果开启了 cuda，就检测 GPU 数；否则当作 1 卡
    if ngpus > 1:
        # 1) 计算原始总交互步数
        total_steps = args.train.step_per_epoch * args.train.epoch
        print(f"Detected {ngpus} GPUs, scaling batch_size, lr and num_processes accordingly")
        # 放大全局 batch_size
        args.train.batch_size *= ngpus
        # 放大学习率
        args.opt.lr *= ngpus
        # 放大采样子进程数
        args.train.num_processes *= ngpus
         # 3) 同时扩展每 epoch 的步数，并重新计算 epoch
        args.train.step_per_epoch *= ngpus
        args.train.epoch = math.ceil(total_steps / args.train.step_per_epoch)

        print(f"=> keep total env steps={total_steps}: now step_per_epoch={args.train.step_per_epoch}, epoch={args.train.epoch}")

    # environments
    train_envs, test_envs = make_envs(args)  # make envs and set random seed

    # network
    actor, critic = build_net(args, device)
    if ngpus > 1:
    # 改为 DDP 包装（替换掉 DataParallelNet） :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
        actor = DistributedDataParallel(actor.to(device),
                                        device_ids=[local_rank],
                                        output_device=local_rank,
                                        find_unused_parameters=True)
        critic = DistributedDataParallel(critic.to(device),
                                        device_ids=[local_rank],
                                        output_device=local_rank,
                                        find_unused_parameters=True)
    else:
        actor = actor.to(device)
        critic = critic.to(device)
    actor_critic = ActorCritic(actor, critic)

    if args.opt.optimizer == 'Adam':
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    elif args.opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(
            actor_critic.parameters(),
            lr=args.opt.lr,
            eps=args.opt.eps,
            alpha=args.opt.alpha,
        )
    else:
        raise NotImplementedError

    lr_scheduler = None
    if args.opt.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.train.step_per_epoch / args.train.step_per_collect) * args.train.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    # RL agent 
    dist = CategoricalMasked
    if args.train.algo == 'PPO':
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            batch_size=args.train.batch_size,
            action_space=train_envs.action_space[0],
            action_scaling=False,            
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            discount_factor=args.train.gamma,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    elif args.algo == 'A2C':    
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    else:
        raise NotImplementedError

    log_path = './logs/' + time_str
    
    is_debug = True if sys.gettrace() else False
    if not is_debug:
        # 仅主进程写日志
        rank = dist_module.get_rank()  if ngpus > 1 else 0
        if rank == 0:
            writer = SummaryWriter(log_path)
            logger = TensorboardLogger(writer,
                                    train_interval=args.log_interval,
                                    update_interval=args.log_interval)
        else:
            logger = LazyLogger()
        # backup the config file, os.path.join(,)
        shutil.copy(args.config, log_path)  # config file
        shutil.copy("model.py", log_path)  # network
        shutil.copy("arguments.py", log_path)  # network
    else:
        logger = LazyLogger()

    # ======== callback functions used during training =========
    def train_fn(epoch, env_step):
        # monitor leraning rate in tensorboard
        # writer.add_scalar('train/lr', optim.param_groups[0]["lr"], env_step)
        pass

    def test_fn(epoch, env_step):
        policy.eval()
        test_collector.reset()
        # 重新收集若干测试 episode
        stats = test_collector.collect(n_episode=10)
        # 支持 dict 或 CollectStats 对象两种情况
        if isinstance(stats, dict):
            ratio = stats.get("ratio", None)
            std   = stats.get("ratio_std", 0.0)
        else:
            ratio = getattr(stats, "ratio", None)
            std   = getattr(stats, "ratio_std", 0.0)
        print(f"[Epoch {epoch}] box utilization ratio={ratio:.4f} (std={std:.4f})")

    def save_best_fn(policy):
        if not is_debug:
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_best.pth'))
        else:
            pass

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_final.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if not is_debug:
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": policy.state_dict(), "optim": optim.state_dict()}, ckpt_path)
            return ckpt_path
        else:
            return None
    
    def watch(train_info):
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=1000)
        ratio = result["ratio"]
        ratio_std = result["ratio_std"]
        total = result["num"]
        print(f"The result (over {result['n/ep']} episodes): ratio={ratio}, ratio_std={ratio_std}, total={total}")
        with open(os.path.join(log_path, f"{ratio:.4f}_{ratio_std:.4f}_{total}.txt"), "w") as file:
            file.write(str(train_info).replace("{", "").replace("}", "").replace(", ", "\n"))

    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = PackCollector(policy, train_envs, buffer)
    test_collector = PackCollector(policy, test_envs)
    
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.train.epoch,
        step_per_epoch=args.train.step_per_epoch,
        repeat_per_collect=args.train.repeat_per_collect,
        episode_per_test=10,
        batch_size=args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=True,      # 确保 test_fn 被调用
        test_fn=test_fn,        # 每个 epoch 后打印 box_ratio :contentReference[oaicite:1]{index=1}
    )
    # run 并获取结果；v1.2 中返回的是 dataclass 而非 dict
    result = trainer.run()
    # 如果后续逻辑仍认为是 dict，可转回
    if not isinstance(result, dict):
        result = vars(result)

    final_save_fn(policy)
    pprint.pprint(f'Finished training! \n{result}')
    watch(result)


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  

    train(args)
