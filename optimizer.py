from collections.abc import Callable, Iterable 
from typing import Optional

from cross_entropy import cross_entropy

import torch 
import math


class AdamW(torch.optim.Optimizer): 
    def __init__(self, 
        params = None, 
        lr=1e-3, 
        beta = (0.9, 0.999),
        weight_decay = 0.01,
        eps=1e-8
    ): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "beta": beta,
            "weight_decay": weight_decay,
            "eps": eps
        }

        self.m = 0 
        self.n = 0

        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None): 
        # Note the closure might be passed in to re-compute the loss!  
        loss = None if closure is None else closure()
        
        # "A parameter group is a single collection of torch.nn.Parameter objects
        # assigns with default params"
        # Each step iterates over each param group (Apply the equation we seek to implement)
        # 
        for group in self.param_groups: 
            
            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            alpha = group["alpha"]
            eps = group["eps"]

            for p in group["params"]:
                grad = cross_entropy(self.params, defaults["batch"])
                alpha_t = alpha * (math.sqrt(1 - b[0]) / (1 - b[1]))

                # Weight decay (L2 Regularization) restricting the values a parameter can take
                # We can measure the distance the parameters are from 0
                # First moment estimate 
                theta = theta - (theta * weight_decay * alpha) 

                self.m = beta[0] * self.m + ((1 - beta[0]) * grad)
                self.v = beta[1] * self.v + ((1 - beta[1]) * grad)

                params = params - ((alpha_t * self.m) / (math.sqrt(self.v) + eplison))

