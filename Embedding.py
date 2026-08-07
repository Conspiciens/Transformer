import torch 


# Token ID -> Vector Space in model_d
class Embedding(torch.nn.Module): 
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None): 
        super(Embedding, self).__init__() 

        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim

        # Embedding Matrix - (vocab_size, d_model)
        # Token IDs - (batch_size, sequence_length)
        self.embedding = torch.nn.Parameter(torch.rand((num_embeddings, embedding_dim))) 
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        ''' 
            Looking up the associated weights with each Token ID 
        ''' 
        return self.embedding[token_ids]
    
    