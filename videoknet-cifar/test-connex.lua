-- test of connex matrix
require 'nnx'

cTable1 = torch.Tensor({{1,1},{2,1},{3,1}})
net = nn.SpatialConvolutionMap(cTable1, 3, 3)
net.bias = net.bias*0

net.weight[1] = torch.ones(3,3)*0
net.weight[2] = torch.ones(3,3)*0
net.weight[3] = torch.ones(3,3)*0

a = torch.ones(3,4,4)
b = net:forward(a)
print(b)

net.weight[1] = torch.ones(3,3)*0
net.weight[2] = torch.ones(3,3)*0
net.weight[3] = torch.ones(3,3)*3

a = torch.ones(3,4,4)
b = net:forward(a)
print(b)


net.weight[1] = torch.ones(3,3)*0
net.weight[2] = torch.ones(3,3)*2
net.weight[3] = torch.ones(3,3)*0

a = torch.ones(3,4,4)
b = net:forward(a)
print(b)

net.weight[1] = torch.ones(3,3)*1
net.weight[2] = torch.ones(3,3)*0
net.weight[3] = torch.ones(3,3)*0

a = torch.ones(3,4,4)
b = net:forward(a)
print(b)


net.weight[1] = torch.ones(3,3)*1
net.weight[2] = torch.ones(3,3)*2
net.weight[3] = torch.ones(3,3)*3

a = torch.ones(3,4,4)
b = net:forward(a)
print(b)

