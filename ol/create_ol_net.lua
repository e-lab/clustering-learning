----------------------------------------------------------------------
-- create a Online Learner neural net based on clustering learning filters
-- E. Culurciello, August 22nd 2012
----------------------------------------------------------------------

require 'nnx'
require 'image'

net_old = torch.load('encoder_orig.net')
kernels1 = torch.load('cifar10-1l-64.t7')
kernels2 = torch.load('cifar10-2l-256.t7')

nk1 = net_old.modules[1].nOutputPlane -- desired number of 1st layer features
nk2 = net_old.modules[4].nOutputPlane -- desired number of 2nd layer features
is1 = net_old.modules[1].kW -- desired 1st layer feature size (supposed square!)
is2 = net_old.modules[4].kW -- desired 2nd layer feature size (supposed square!)

kernels1 = kernels1[{{1,nk1}}]
kernels2 = kernels2[1][{{1,nk2}}] -- just take the 1st 'nk' kernels and use these

osz1 = torch.sqrt(kernels1:size(2)) -- (supposed square!)
osz2 = torch.sqrt(kernels2:size(2)) -- (supposed square!)

kernels1 = kernels1:reshape(nk1, osz1, osz1)
kernels2 = kernels2:reshape(nk2, osz2, osz2)

weight1 = torch.Tensor(nk1,is1,is1)
weight2 = torch.Tensor(nk2,is2,is2)

for i=1, nk1 do 
   weight1[i] = image.scale(kernels1[i], is1, is1)
end
for i=1, nk2 do 
   weight2[i] = image.scale(kernels2[i], is2, is2)
end


-- initialize 1st layer parameters to learned filters:
for i=1, nk1 do   
   net_old.modules[1].weight[i] = weight1[i]:type('torch.FloatTensor')
end
net_old.modules[1].bias = net_old.modules[1].bias *0
-- initialize 2nd layer parameters to learned filters:
for i=1, nk2 do   
   net_old.modules[4].weight[i] = weight2[i]:type('torch.FloatTensor')
end
net_old.modules[4].bias = net_old.modules[4].bias *0


torch.save('encoder_new.net', net_old)