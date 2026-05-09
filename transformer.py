import numpy as np

import torch
from torch import Tensor

HEIGHT = 8 
MATRIX_SIZE = 512

class Transformer: 
    def __init__(self): 
        self.weights = torch.empty(MATRIX_SIZE, MATRIX_SIZE) 

        torch.nn.init.uniform_(self.weights) 

    def forward(self, x: torch.Tensor) -> None: 
        self.weights = x @ self.weights
        self.weights = torch.reshape(self.weights, (1, 8, 64))         
        self.weights = torch.transpose(self.weights, 0, 1)

        test = MultiHeadAttention().forward(x, self.weights)  
        print(test)
        
        # Multihead Attention * Weights (Output)

class Encoder: 
    pass

class Decoder: 
    pass

class DotProductAttention: 
    def __init__(self, query: Tensor, key: Tensor, value: Tensor): 
        self.q = query 
        self.k = key 
        self.v = value

    def forward(self): 
        x = (self.q @ self.k.T)
        x = (x / torch.sqrt(x.shape[2])) @ self.v

        return x 

''' 
    Multi-Head Attention has multiple heads
''' 
class MultiHeadAttention: 
    def forward(self, x: torch.Tensor, w: torch.Tensor): 
        print(f"Size Input: {x.size()}") 
        print(f"Weight Input: {w.size()}") 
    
        W_q = w @ x
        W_k = w @ x 
        W_v = w @ x

        return DotProductAttention(W_q, W_k, W_v).forward() 


if __name__ == '__main__': 
    transformer = Transformer()

    t1 = torch.randn(1, 64)
    transformer.forward(t1)
    
