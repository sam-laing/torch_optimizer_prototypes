import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import Tensor
import math


from .types import Betas2, OptFloat, OptLossClosure, Params, State
ParamGroup = Dict[str, Any]

import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import Tensor
import math

ParamGroup = Dict[str, Any]

class NestedMA(Optimizer):
    r"""Implements NestedMA Optimizer Algorithm.

    just trying something out
    
    """
    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        eps: float = 1e-8,
        betas: Betas2 = (0.9, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        lp: float = 2.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self._weight_decouple = weight_decouple
        self.lp = lp
    
    
    def _get_options(
        self, param_group: ParamGroup, param_shape: Tuple[int, ...]
    ) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment
    
    def _get_lr(self, param_group: ParamGroup, param_state: State) -> float:
        return param_group["lr"]

    

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("NestedMA doesn't support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["nested_exp_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg_sq = state["exp_avg_sq"]
                nested_exp_ma = state["nested_exp_ma"]

                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Update second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected second moment
                bias_correction2 = 1 - beta2 ** state["step"]
                corrected_exp_avg_sq = exp_avg_sq.div(1 - bias_correction2)

                # Update nested moving average
                nested_exp_ma.mul_(beta1).addcdiv_(grad, corrected_exp_avg_sq.sqrt().add_(group["eps"]), value=1 - beta1)

                # Compute bias-corrected first moment
                bias_correction1 = 1 - beta1 ** state["step"]
                corrected_nested_ma = nested_exp_ma   #.div(1 - bias_correction1)

                # Apply weight decay if specified
                if group["weight_decay"] != 0:
                    #p.data = p.data + group["weight_decay"] * torch.norm(p.data, p=self.lp)
                    """
                    # decoupled weight decay as in AdamW but good ol' Lp norm easily possible too 
                    if self._weight_decouple:
                        p.data.add_(group["weight_decay"] * torch.norm(p.data, p=self.lp))
                    else:
                        p.data.add_(group["weight_decay"] * group["lr"] * torch.norm(p.data, p=self.lp))
                    """
                    pass

                # Update parameters
                step_size = group["lr"] * corrected_nested_ma
                p.data.add_(-step_size)

        return loss