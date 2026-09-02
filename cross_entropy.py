import torch

def cross_entropy(x: torch.Tensor, t: torch.Tensor): 
    ''' 
        @params: 
            x: (batch_size, vocab_size)
            t: (batch_size) with the index of the correct class (target that we want the answer for)
    ''' 
    c = torch.max(x, dim=-1, keepdim=True).values
    x = x - c

    print(x)
    print(x.size())
    # torch.exp(x[...,t]) (8, 5)
    # the issue with the code above is it triggers "advanced index rules" so instead of [8, 1] it's [8, 8]
    # cause for every row coord it pairs with t
    # However, torch.arange(x.size(0)) gives the row idx. Results in [0, t[0]], [1, t[1]]
    ans = x[torch.arange(x.size(0)),t]
    denominator = torch.sum(torch.exp(x), dim=-1)
    loss = -ans + torch.log(denominator)

    # Across the entire batch means across all the batches within the matrix
    return torch.mean(loss)
    

