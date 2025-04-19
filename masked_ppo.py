# masked_ppo.py

from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch
from tianshou.policy.modelfree.ppo import PPOPolicy


class MaskedPPOPolicy(PPOPolicy):
    """在 PPOPolicy 基础上增加 Mask 支持，并示例如何保留跨 step 的内部状态。"""

    def __init__(
        self,
        *,
        actor: nn.Module,
        critic: nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Any,
        # 以下参数都直接透传给父类 PPOPolicy
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        max_grad_norm: Optional[float] = None,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        # 1) 调用父类构造函数，所有 update/learn 逻辑由官方实现
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            max_grad_norm=max_grad_norm,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            **kwargs,
        )
        # 2) 在 Policy 对象上保留一个跨 step 的计数器，示例用
        self.my_counter = 0
        self._batch = batch_size

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """重写 forward，只做 Mask 及内部状态更新，返回格式与 PPOPolicy.forward 相同。"""
        # 1) 用 actor 得到 logits 和下一个 hidden state
        logits, hidden = self.actor(batch.obs, state=state)
        # 2) 从 obs.mask 构造 boolean mask（假设 obs.mask 是可以 as_tensor 的）
        mask = torch.as_tensor(batch.obs.mask, dtype=torch.bool, device=logits.device)
        # 3) 调用 dist_fn 构造分布，传入 logits 和 mask
        #    dist_fn 接口依赖于你定义的 Distribution，确保支持 masks 参数
        if isinstance(logits, tuple):
    # 假如你的 actor 返回 (logits, aux)，就这样
            dist = self.dist_fn(*logits, masks=mask)
        else:
            dist = self.dist_fn(logits=logits, masks=mask)
        # 4) 依据 deterministic_eval 决定是 sample 还是 argmax
        if self.deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = dist.logits.argmax(dim=-1)
            else:  # continuous
                act = logits
        else:
            act = dist.sample()
        # 5) 更新内部计数器（示例），你也可以在这里做更多状态记录
        self.my_counter += int(act.numel())
        # 6) 返回 Batch：key 必须包含 act, logits, dist, state
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def get_counter(self) -> int:
        """外部可以调用这个接口来查看累计的动作数。"""
        return self.my_counter
