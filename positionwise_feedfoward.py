import torch 

# Relu(x) usually was used 
# 
# 

class positionwise_feedfoward(torch.nn.Module): 
    def __init__(self, d_model, d_ff): 
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff 


        self.W1 = torch.nn.Parameter(torch.rand(
            d_ff, d_model
        ))
        self.W2 = torch.nn.Parameter(torch.rand(
            d_model, d_ff 
        ))
        self.W3 = torch.nn.Parameter(torch.rand(
            d_ff, d_model 
        ))



    def forward(self, x: torch.Tensor): 
        lt = x @ self.W1.T         
        silu = torch.nn.functional.sigmoid(lt) * lt

        return (silu * (x @ self.W3.T)) @ self.W2.T

        
        
        