import torch 

def gradient_clipping(p, M): 
    ''' 

        Note: Seems that several gradients can be None, this is because when beginning to train it hasn't been calculated in the backward pass
    
    ''' 
    e = 10e-6

    l2_norm = 0
    for parameter in p: 
        if parameter.grad is None: 
            continue 
        l2_norm_param = torch.square(parameter.grad)
        l2_norm += torch.sum(l2_norm_param)
        
    l2_norm = torch.sqrt(l2_norm)

    if l2_norm > M: 
        l2_norm = l2_norm + e
        scale = M / l2_norm 

        for parameter in p: 
            if parameter.grad is None: 
                continue 
            parameter.grad = parameter.grad * scale
                

    




    
