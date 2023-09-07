import torch

class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in,fan_out)) * fan_in**-0.5 # kaiming
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self,x):
        self.out = x @ self.weight # only for future reference
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight,self.bias]
#-------------------------------------------------------  
class BatchNorm1d:
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        #parameters (trained with backprop) (shift and scale)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #buffers (trained with running "momentum update")
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self,x):
        #calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim,keepdim=True)
            xvar = x.var(dim,keepdim=True,unbiased = True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x-xmean)/torch.sqrt(xvar+self.eps)
        self.out = self.gamma*xhat + self.beta # only for future reference
        #update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*xmean
                self.running_var = (1-self.momentum)*self.running_var + self.momentum*xvar
        return self.out
            
    
    def parameters(self):
        return [self.gamma, self.beta]
#-------------------------------------------------------
class Tanh:
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
#-------------------------------------------------------
class Embedding:
    
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings,embedding_dim))
    
    def __call__(self,IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
#-------------------------------------------------------
class FlattenConsecutive: # different from pytorch
    
    def __init__(self,n):
        self.n = n
        
    def __call__(self,x):
        B,T,C = x.shape
        x = x.view(B,T//self.n,C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1) # delete dim at 1 if 1
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
########################################################

class Sequential:
    
    def __init__(self,layers):
        self.layers = layers
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    