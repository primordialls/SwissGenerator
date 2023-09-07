import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import nn
from torch import tensor as tt

# read in all the words
words = open('cities.txt', 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

#shuffle up words
import random
random.seed(42)
random.shuffle(words)

# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
    X, Y = [], []
  
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

torch.manual_seed(42)

n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of the MLP

model  = nn.Sequential([
    nn.Embedding(vocab_size,n_embd),
    nn.FlattenConsecutive(2), nn.Linear(n_embd*2,n_hidden,bias=False), nn.BatchNorm1d(n_hidden), nn.Tanh(),
    nn.FlattenConsecutive(2), nn.Linear(n_hidden*2,n_hidden,bias=False), nn.BatchNorm1d(n_hidden), nn.Tanh(),
    nn.FlattenConsecutive(2), nn.Linear(n_hidden*2,n_hidden,bias=False), nn.BatchNorm1d(n_hidden), nn.Tanh(),
    nn.Linear(n_hidden,vocab_size),
])

#parameter init

with torch.no_grad():
    #last layer: make less confident
    model.layers[-1].weight *= 0.1 #refer to W2 from before

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

#fixing batchnorm
e = torch.randn(32,4,68)
emean = e.mean(0,keepdim=True) # 1,4,68
evar = e.var(0,keepdim=True) # 1,4,68
ehat = (e-emean)/torch.sqrt(evar+1e-5) # 32,4,68

# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps+1):
    
    #minibatch construct
    ix = torch.randint(0,Xtr.shape[0],(batch_size,))
    Xb, Yb = Xtr[ix],Ytr[ix] #batch X,Y
    
    #forward pass
    logits = model(Xb)
    print(Yb.shape)
    loss = F.cross_entropy(logits,Yb)
    
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i<(max_steps*0.75) else 0.01
    for p in parameters:
        p.data += -lr*p.grad
        
    #track stats
    if (i%1000==0): 
        print(f"\r{i/max_steps:.1%}" + f" {loss.item():.4f} ",end="")
    lossi.append(loss.log10().item())

    #break


#eval model
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
         'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits,y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

#sample
for _ in range(20):
    
    out = []
    star = "."
    context = [0] * (block_size-1) + [stoi[star]]
    
    while True:
        # forward pass the neural net
        logits = model(tt([context]))
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break
    print((star if star!="." else "") + "".join(itos[i] for i in out[:-1]))