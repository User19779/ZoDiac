import torch

x1=torch.tensor(
    [1.0,2.0,3.0], requires_grad=True
)
W = torch.tensor(
    [[1.0,2.0,0.0,],
    [6.0,0.0,4.0,],
    [3.0,5.0,1.0,],]
)
x2 = torch.matmul(W,x1)
x3=x2[1]

grad = torch.autograd.grad(outputs=x3, inputs=x1)
print(grad)
