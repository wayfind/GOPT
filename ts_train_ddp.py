from datetime import datetime
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
import torch.distributed as dist
import tianshou as ts
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, DataParallelNet
from tianshou.trainer import onpolicy_trainer
import model
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from mycollector import PackCollector

def setup_logging(args, ngpus):
    # 检测是否在调试器下
    is_debug = True if sys.gettrace() else False
    # 计算当前进程 rank（单卡默认为 0）
    rank = dist_module.get_rank() if ngpus > 1 else 0

    # 默认都给这两个，防止未定义错误
    writer = None
    logger = LazyLogger()
    log_path = None

    # 只有主进程且非调试模式时真正初始化 TensorBoard
    if not is_debug and rank == 0:
        # 1) 生成一个带微秒的唯一目录名
        ts = datetime.now().strftime('%Y.%m.%d-%H-%M-%S-%f')
        name = (
            f"{args.env.id}_"
            f"{args.env.container_size[0]}-{args.env.container_size[1]}-{args.env.container_size[2]}_"
            f"{args.env.scheme}_{args.env.k_placement}_"
            f"{args.env.box_type}_{args.train.algo}_"
            f"seed{args.seed}_{args.opt.optimizer}_"
            f"{ts}"
        )
        log_base = "logs"
        log_path = os.path.join(log_base, name)

        # 2) 确保父目录存在，然后创建自身目录
        os.makedirs(log_base, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # 3) 初始化 SummaryWriter & TensorboardLogger
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval,
        )

        # 4) 备份配置和关键脚本
        for fname in (args.config, "model.py", "arguments.py"):
            try:
                shutil.copy(fname, log_path)
            except FileNotFoundError:
                # 若某个文件不存在，可根据需要忽略或报 warn
                print(f"[rank {rank}] Warning: cannot backup {fname}")

    return writer, logger, log_path

def make_envs(args):
    """创建训练和测试环境。"""
    train_envs = ts.env.SubprocVectorEnv(
        [
            lambda: gym.make(
                args.env.id,
                container_size=args.env.container_size,
                enable_rotation=args.env.rot,
                data_type=args.env.box_type,
                item_set=args.env.box_size_set,
                reward_type=args.train.reward_type,
                action_scheme=args.env.scheme,
                k_placement=args.env.k_placement,
            )
            for _ in range(args.train.num_processes)
        ]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [
            lambda: gym.make(
                args.env.id,
                container_size=args.env.container_size,
                enable_rotation=args.env.rot,
                data_type=args.env.box_type,
                item_set=args.env.box_size_set,
                reward_type=args.train.reward_type,
                action_scheme=args.env.scheme,
                k_placement=args.env.k_placement,
            )
            for _ in range(1)
        ]
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    return train_envs, test_envs


def init_distributed(args):
    """初始化分布式训练（如果使用多个 GPU）。"""
    if args.cuda and torch.cuda.device_count() > 1:
        dist.init_process_group(
            backend="nccl",  # 使用 NCCL 进行 GPU 通信
            init_method="env://",  # 使用环境变量进行设置
            rank=args.local_rank,  # 当前进程的秩
            world_size=args.world_size,  # 总进程数
        )
        torch.cuda.set_device(args.local_rank)  # 设置当前进程的设备
    elif args.cuda and torch.cuda.device_count() == 1:
        torch.cuda.set_device(args.device)


def cleanup_distributed():
    """清理分布式训练。"""
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


def get_device(args):
    """获取正确的设备 (GPU 或 CPU)"""
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)  # 保持默认
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")
    return device


def to_distributed(net, args, device):
    """如有必要，包装网络以进行分布式训练。"""
    if args.cuda and torch.cuda.device_count() > 1:
        net = DataParallelNet(net, device_ids=[args.local_rank]).to(device)
    return net


def build_net(args, device):
    """构建 actor 和 critic 网络。"""
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
        place_gen=args.env.scheme,
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
    """训练 RL 智能体。"""
    date = time.strftime(r"%Y.%m.%d-%H-%M-%S", time.localtime(time.time()))
    time_str = (
        args.env.id
        + "_"
        + str(args.env.container_size[0])
        + "-"
        + str(args.env.container_size[1])
        + "-"
        + str(args.env.container_size[2])
        + "_"
        + args.env.scheme
        + "_"
        + str(args.env.k_placement)
        + "_"
        + args.env.box_type
        + "_"
        + args.train.algo
        + "_"
        + "seed"
        + str(args.seed)
        + "_"
        + args.opt.optimizer
        + "_"
        + date
    )

    # 初始化分布式训练（如果适用）
    init_distributed(args)

    # 获取设备
    device = get_device(args)

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # 环境
    train_envs, test_envs = make_envs(args)  # 创建环境并设置随机种子

    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count() if args.cuda and torch.cuda.is_available() else 1
    print(f"Number of GPUs: {num_gpus}")

    # 动态调整 batch_size
    if num_gpus > 1:
        args.train.batch_size = args.train.batch_size * num_gpus
        print(f"Adjusted batch_size to {args.train.batch_size} for {num_gpus} GPUs")
    else:
        print(f"Using batch_size: {args.train.batch_size} for single GPU")

    # 网络
    actor, critic = build_net(args, device)
    actor = to_distributed(actor, args, device)  # 包装
    critic = to_distributed(critic, args, device)  # 包装
    actor_critic = ActorCritic(actor, critic)  # 这仅用于优化器

    if args.opt.optimizer == "Adam":
        optim = torch.optim.Adam(
            actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps
        )
    elif args.opt.optimizer == "RMSprop":
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
        # 线性衰减学习率到 0
        max_update_num = (
            np.ceil(args.train.step_per_epoch / args.train.step_per_collect)
            * args.train.epoch
        )
        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # RL 智能体
    dist = CategoricalMasked
    if args.train.algo == "PPO":
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=args.train.gamma,
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler,
        )
    elif args.train.algo == "A2C":
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler,
        )
    else:
        raise NotImplementedError

    log_path = "./logs/" + time_str

    is_debug = True if sys.gettrace() else False
    if not is_debug:
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer=writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval,
        )
        # 备份配置文件
        shutil.copy(args.config, log_path)  # 配置文件
        shutil.copy("model.py", log_path)  # 网络
        shutil.copy("arguments.py", log_path)  # 参数
    else:
        logger = LazyLogger()

    # ======== 训练期间使用的回调函数 =========
    def train_fn(epoch, env_step):
        # monitor leraning rate in tensorboard
        # writer.add_scalar('train/lr', optim.param_groups[0]["lr"], env_step)
        pass

    def save_best_fn(policy):
        if not is_debug:
            torch.save(
                policy.state_dict(), os.path.join(log_path, "policy_step_best.pth")
            )
        else:
            pass

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy_step_final.pth"))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if not is_debug:
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "env_step": env_step,
                    "gradient_step": gradient_step,
                },
                ckpt_path,
            )
            return ckpt_path
        else:
            return None

    def watch(train_info, policy, test_envs, args, log_path):  # 添加参数
        print("设置测试环境...")
        policy.eval()
        test_envs.seed(args.seed)
        print("测试智能体...")
        test_collector.reset()
        result = test_collector.collect(n_episode=1000)
        ratio = result["ratio"]
        ratio_std = result["ratio_std"]
        total = result["num"]
        print(
            f"The result (over {result['n/ep']} episodes): ratio={ratio}, ratio_std={ratio_std}, total={total}"
        )
        with open(
            os.path.join(log_path, f"{ratio:.4f}_{ratio_std:.4f}_{total}.txt"), "w"
        ) as file:
            file.write(str(train_info).replace("{", "").replace("}", "").replace(", ", "\n"))

    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = PackCollector(policy, train_envs, buffer, device=device)  # add device
    test_collector = PackCollector(policy, test_envs)

    # 训练器
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.train.epoch,
        step_per_epoch=args.train.step_per_epoch,
        repeat_per_collect=args.train.repeat_per_collect,
        episode_per_test=10,  # args.test_num,
        batch_size=args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        # episode_per_collect=args.episode_per_collect,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,
    )

    final_save_fn(policy)
    if args.local_rank == 0:  # only print once.
        pprint.pprint(f"Finished training! \n{result}")
        watch(result, policy, test_envs, args, log_path)  # pass new args
    cleanup_distributed()  # add cleanup


if __name__ == "__main__":
    registration_envs()
    args = arguments.get_args()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps

    train(args)

