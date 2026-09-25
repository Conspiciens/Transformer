import random
import numpy as np
import torch 

def data_loader(token_ids, batch_size, context_length, device): 
    '''
        Loading data is via guessing a i value that's <= n - m, overlapping doesn't matter  
    ''' 

    random_idxs = np.random.choice(len(token_ids) - context_length, size=(batch_size,))

    predict_batches = np.stack([token_ids[idx: idx + context_length] for idx in random_idxs])
    answer_batches = np.stack([token_ids[idx + 1: idx + 1 + context_length] for idx in random_idxs])

    predict_batches = torch.from_numpy(predict_batches)
    answer_batches = torch.from_numpy(answer_batches)
    
    return (predict_batches, answer_batches)

    # Both Tensors should have (batch_size, context_len)
    