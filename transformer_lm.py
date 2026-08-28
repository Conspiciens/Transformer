import torch 

from Embedding import Embedding
from Linear import Linear
from transformer_block import TransformerBlock
from softmax import softmax 
from rmsnorm import rmsnorm 

class Transformer(torch.nn.Module): 
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        vocab_size: int, 
        context_length: int,
        num_layers: int
    ): 
        super().__init__()
        # (self, num_embeddings: int, embedding_dim: int, device=None, dtype=None): 
        self.d_model = d_model
        self.num_heads = num_heads 
        self.d_ff = d_ff 
        self.vocab_size = vocab_size 
        self.context_length = context_length 
        self.num_layers = num_layers
        
        # Reverse, since we want the output of probabilites
        self.linear = Linear(d_model, vocab_size)

        # Num layers are related to the number of Transformer blocks 
        # Embedding Matrix - (vocab_size, d_model)
        self.token_embedding = Embedding(vocab_size, d_model)

        # Store the blocks in a list
        self.blocks = num_layers * [None] 
        for i in range(num_layers): 
            # d_ff is for the inner layers 
            self.blocks[i] = TransformerBlock(d_model, num_heads, d_ff)

        # (d_model, d_model)
        self.final_rmsnorm = rmsnorm(d_model)
        

    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor: 
        ''' 
            @parameters + self 
                token_embedding: (batch_size, seq_len, d_model)
                TransformerBlock: (seq_len, d_model) ? 
                

            @return 
                (batch_size, seq_len, vocab_size)
                
        ''' 
    
        x = self.token_embedding.forward(in_indices)
        for i in range(self.num_layers): 
            x = self.blocks[i].forward(x)

        print(f"\n\nTransformer loop: {x.size()}")
        # RMSNorm
        x = self.final_rmsnorm.forward(x)
        
        print(f"Norm: {x.size()}")
        # Linear  
        x = self.linear.forward(x)

        print(f"Linear: {x.size()}")
        return x
        # Softmax
        # return softmax(x, -1)

