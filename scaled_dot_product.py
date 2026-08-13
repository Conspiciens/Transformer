import torch 
import math 

from softmax import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask) -> torch.Tensor: 
    ''' 
        Scaled dot product attention 
    ''' 

    dim = Q.size()[-1]
    print(f"Key: {torch.transpose(K, 1, 2).size()}")
    print(f"Q: {Q.size()}")

    num = Q @ torch.transpose(K, -2, -1)
    dem = math.sqrt(dim)

    total = num / dem

    # We use a mask to give attention from Query to Key, or even remove it 
    # When False we * by negative -inf to reduce attention
    # After doing softmax the attention queries should up to 1, while false should remain 0 
    mask = ~mask
    mask = mask.to(torch.int32)
    mask = mask * -torch.inf
    mask = torch.nan_to_num(mask, 0)

    total = total + mask

    return softmax(total, -1) @ V
