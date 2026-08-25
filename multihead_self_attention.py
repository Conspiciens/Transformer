import torch

from scaled_dot_product import scaled_dot_product_attention
from rope import rope 

class multihead_self_attention(torch.nn.Module): 
    def __init__(self, d_model: int, num_heads: int): 
        super().__init__()

        self.Q = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.K = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.V = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.O = torch.nn.Parameter(torch.randn(d_model, d_model))

        print(f"d_model: {d_model}")

        self.d_k = d_model // num_heads

        self.num_heads = num_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        Q_x = x @ self.Q.T
        K_x = x @ self.K.T
        V_x = x @ self.V.T

        seq_len = x.size()[1]
        print(f"Q_x: {Q_x.size()}")

        rope_c = rope(10000.0, self.d_k, seq_len)
        # 
        # [4, 12, 64]
        # [batch_size, head_num, seq_len, d_k]
        # d_k the number of dimension for keys & queries (Dimension of keys & queries)
        # Want each head to point towards tokens from across the entire features
        # 
        # [3072]
        # 

        # torch.reshape actually corrupts the features, requires a transpose from (1, 2)
        # reshape reads contingous memory [4, 12, 16] so it actually just reading the same token features, when we need features
        # across all tokens
        # We keep the [batch_size, seq_len, head_num, d_k] but then transpose
        # [12, 16]
        #  | 16 |
        # [
        #  12 
        # ]
        # Theory: [A0, A1, A2, A3, B0, B1, B2, B3]
        # reshape: [A0, A1] [A2, A3]
        Q_reshape = torch.reshape(Q_x, (x.size()[0], seq_len, self.num_heads, self.d_k))
        K_reshape = torch.reshape(K_x, (x.size()[0], seq_len, self.num_heads, self.d_k))
        V_reshape = torch.reshape(V_x, (x.size()[0], seq_len, self.num_heads, self.d_k))

        Q_reshape = Q_reshape.transpose(1, 2)
        K_reshape = K_reshape.transpose(1, 2)
        V_reshape = V_reshape.transpose(1, 2)


        positions = torch.arange(seq_len, device=x.device)
        print(f"SequuezE?: {Q_reshape[:, :, :, 0].size()}")

        # We only need one seq_len 1d vector
        # All the tokens are going to be taking up the same position

        Q_reshape = rope_c.forward(Q_reshape, positions)
        K_reshape = rope_c.forward(K_reshape, positions)

        # [batch_size, heads, seq_len (tokens), features]
        # mask = Q_reshape @ torch.transpose(K_reshape, -2, -1)
        # mask = torch.tril(mask, diagonal=0).bool()

        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).bool()
        print(f"Heads: {self.num_heads}")
        att = scaled_dot_product_attention(Q_reshape, K_reshape, V_reshape, mask)

        att = att.transpose(1, 2)
        # Contiguous 
        # Reshape
        att = att.contiguous().reshape(x.size()[0], seq_len, self.d_model)

        # Merge the heads
        print(f"ATT: {att.size()}")
        

        return att @ self.O.T
