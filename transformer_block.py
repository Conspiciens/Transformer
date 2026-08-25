import torch 

from multihead_self_attention import multihead_self_attention
from rmsnorm import rmsnorm
from positionwise_feedfoward import positionwise_feedfoward

class TransformerBlock(torch.nn.Module): 
    def __init__(self, d_model: int, num_heads: int, d_ff: int): 
        super().__init__()
        self.RMSnorm1 = rmsnorm(d_model) 
        self.RMSnorm2 = rmsnorm(d_model)
        self.MultiHeadSelfAttention = multihead_self_attention(d_model, num_heads)
        # d_ff feedfoward neural network 
        self.PositionwiseFF = positionwise_feedfoward(d_model, d_ff) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        ''' 
            first block: multi head self attention then RMSnorm(x) Add x 
            second block: 
        ''' 
        mhsf = x + self.MultiHeadSelfAttention.forward(self.RMSnorm1(x))
        return mhsf + self.PositionwiseFF.forward(self.RMSnorm2(mhsf))
