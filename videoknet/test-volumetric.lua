
require 'nnx'
require 'image'
      
poolsize = 2
nk1=32
nk2 = 32
fanin1 = 1
fanin2 = 4
is1 = 7
is2 = 7

net_new = nn.Sequential()
-- usage: VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
net_new:add(nn.VolumetricConvolution(3, nk1, 2, is1, is1))
net_new:add(nn.Sum(2))
net_new:add(nn.Tanh())
net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
-- 2nd layer:
net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
net_new:add(nn.Tanh())
net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 


-- Volumetric Convolutions tests:
inp = torch.Tensor(3,2,64,64) -- 3 planes, 2 frames of 64x64 each
out = net_new:forward(inp) -- should be 32 x 29 x 29

print('VolumetricConv out size:', net_new.modules[2].output:size())
print('Overall net out size:', out:size())

