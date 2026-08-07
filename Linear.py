import torch 

import torch.nn.functional as F

# Linear Transformation (Every Linear transfromation is a matrix transformation)
#  must satisfy two things 
#    1) T(u + v) = A(u + v) = Au + Av = T(u) + T(v)
#    2) T(cu) = A(cu) = c(Tu)
#   y = Wx

class Linear(torch.nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        super(Linear, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.W = torch.nn.Parameter(torch.empty((self.out_features, self.in_features)))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        # x @ self.W.T vs self.W.T @ x 
        # (4, 12, 64) @ (128, 64) vs (128, 64) @ (4, 12, 64)
        # (256, 64) @ (128, 64) vs (128, 64) @ ()
        

        # x = (4, 12, 64) W = (128, 64) but T = (64, 128)
        # x
        return x @ self.W.T
        