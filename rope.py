import torch 

class rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        super().__init__()
        self.theta = theta 
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # K is a vector that controls it's frequency scale 
        # frequency scale is 

        k = torch.arange(0, d_k, 2).float()
        k = (k / d_k)
    
        self.angles = 1 / torch.pow(self.theta, k)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        '''
        ''' 
        # q'1 = q1 * cos(theta) - q2 * sin(theta)
        # q'2 = q2 * sin(theta) - q3 * cos(theta)
        # q1 is x-axis 
        # q2 is y-axis 
        # Each token is represented as a vector, bunch of floating point numbers

        # Set up Ri
        # Torch outer creates a matrix of (token_position, self.angles) via merging
        true_angles = torch.outer(token_positions, self.angles)
        # Repeat interleave? [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        # we do this cause [theta 0, theta 1, theta 2] -> [theta 0, theta 0, theta 1, theta 1]
        # dim = -1 -> last dimension
        true_angles = torch.repeat_interleave(true_angles, 2, dim=-1)

        # x1: [q1, q3] 
        # x2: [q2, q4]
        x1 = x[..., 0::2] 
        x2 = x[..., 1::2] 

        # Conctecating every two values 
        negative_x = torch.stack((-x2, x1), dim=-1).flatten(-2)
        # [-q2, q1, -q3, q4]
        # sin() 
        # [q1, q2, q3, q4]
        # cos

        R = (negative_x * torch.sin(true_angles)) + (x * torch.cos(true_angles))

        return R


        