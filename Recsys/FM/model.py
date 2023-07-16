import torch 
import torch.nn as nn 

class Factorization(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Factorization, self).__init__()
        self.factor = nn.Parameter(torch.randn(input_dim, output_dim))
        
    def forward(self, x):
        square_of_sum = torch.pow(torch.matmul(x, self.factor),2)
        sum_of_square = torch.matmul(torch.pow(x,2), torch.pow(self.factor,2))
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
    
class FM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FM, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.factorization = Factorization(input_dim, output_dim)
        
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.factorization.factor, std=0.01)
    
    def forward(self, x):
        x = self.linear(x).squeeze() + self.factorization(x)
        
        return x