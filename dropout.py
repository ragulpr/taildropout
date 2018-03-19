class PartialLinear(nn.Linear):
    r"""A linear layer that only uses the first k input features.
    
    Equivalent to dropping out the last N-k input features.
    Equivalent to setting last N-k input features to zero.
    Equivalent to setting last N-k weight-columns to zero.
        
    Theoretically faster than dropout. 
    Can be sampled from discretized Beta to force inclusion probability towards equal.
    
    Examples::
        >>> linear = PartialLinear(3, 2)
        >>> input = Variable(torch.ones(2, 3))
        >>> input[:,2] = 1000
        >>> output = linear(input,2)
        >>> print(output)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(PartialLinear, self).__init__(in_features, out_features, bias=True)
        
    def forward(self,input,k):
        return F.linear(input[...,:k], self.weight[...,:k], self.bias)

class ContiguousDropoutLinear(nn.Module):
    r"""During training, randomly zeroes 
    """
    def __init__(self, ):
        super(ContiguousDropoutLinear, self).__init__()
        self.linear = PartialLinear(nn.Linear)
    
    def forward(self, input):
        if self.training:
            n_features =  input.shape[self.feature_dim]
            dropout_start = torch.LongTensor(input.shape[-self.feature_dim]).random_(n_features)
            input = 
            
        return F.dropout(input, self.p, self.training, self.inplace)
    