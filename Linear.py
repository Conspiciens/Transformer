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
        print("Weights:" + str(self.W.size()))
        print("Vector: " + str(x.size()))
        
        self.out_features = x @ self.W.T
        return self.out_features
        