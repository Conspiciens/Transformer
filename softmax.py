import torch

# Softmax is invarient to adding any constants (c)
# exp() is e^x

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: 

    c = torch.max(x, dim=-1, keepdim=True).values
    x = x - c

    total = torch.sum(torch.exp(x), dim=-1, keepdim=True)

    return torch.exp(x) / total
