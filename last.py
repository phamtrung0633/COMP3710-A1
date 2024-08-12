import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
x = torch.Tensor(X)
y = torch.Tensor(Y)
c = torch.complex(x, y) #important!

z = torch.zeros_like(c)
ns = torch.zeros_like(c)
z = z.to(device)
c = c.to(device)
ns = ns.to(device)
for i in range(200):
    z_ = (torch.abs(z.real) + 1j * torch.abs(z.imag)) ** 2 + c
    #Have we diverged with this new value?
    not_diverged = torch.abs(z_) < 4.0
    #Update variables to compute
    ns += not_diverged
    z = z_
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))
print(ns.cpu().numpy())
def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()