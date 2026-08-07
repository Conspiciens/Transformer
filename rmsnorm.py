import torch

# Post-Norm Transformer: Layer normalizatin is applied after the sub-layer output
# Pre-Norm: Layer normalization is applied before Multi-Head Attention & positionwise feedforward 
#   Note: "Clean residual stream without any normalization going from the input embeddings to the output embeddings"


# RMSNorm(ai) = (ai / RMS(a)) * gi
# RMS(a) = sqrt(1/d_model * sum(a + e))

class rmsnorm(torch.nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None): 
        super(rmsnorm, self).__init__() 
        self.d_model: int  = d_model
        self.eps: float = eps 

        self.W = torch.nn.Parameter(torch.rand(d_model))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1.0, a=-3.0, b=3.0)


    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        ''' 
            x: (batch_size, sequence_length, d_model)
        ''' 
        in_dtype = x.dtype
        # Use float32 to avoid overflowing 
        x = x.to(torch.float32)

        a = x[2]

        # Calculate Root Mean Square first? 
        # Keep iterating through row after row
        # dim = -1 collapse the last axis for summation 
        # dim = -1 
        # keepdim = True Maintains the last dimension
        rms = torch.sqrt((torch.sum(torch.square(x), dim=-1, keepdim=True) + self.eps) * (1 / self.d_model))

        # Then solve for Root Mean Square Normalization 
        result = torch.div(x, rms) * self.W

        return result.to(in_dtype)