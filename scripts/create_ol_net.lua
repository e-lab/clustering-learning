----------------------------------------------------------------------
-- create a Online Learner neural net based on clustering learning filters
-- E. Culurciello, August 22nd 2012
----------------------------------------------------------------------

require 'nnx'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Generates a new online-learner network')
cmd:text()
cmd:text('Options')
cmd:option('-archn', 0, 'create new architecture or leave old one intact?')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:


net_orig = torch.load('encoder_orig.net') -- this is the original OL network renamed - a copy has to be in this folder!
--image.display{image=net_orig.modules[1].weight, zoom=4, padding=2, nrow =4}  -- to visualize filters
--image.display{image=net_orig.modules[4].weight[{{1,128}}], zoom=4, padding=2, nrow =16}
--image.display{image=net_orig.modules[7].weight[{{1,128}}], zoom=4, padding=2, nrow =16}


kernels1 = torch.load('berkeley56x56-1l-64.t7')
kernels2 = torch.load('berkeley56x56-2l-256.t7')

--image.display{image=kernels1:reshape(nk1,is1,is1), padding=2, symmetric=true, 
--         zoom=2, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
--         
--image.display{image=kernels2:reshape(nk2,is2,is2), padding=2, symmetric=true, 
--         zoom=2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'}

nk1 = net_orig.modules[1].nOutputPlane -- desired number of 1st layer features
nk2 = net_orig.modules[4].nOutputPlane -- desired number of 2nd layer features
is1 = net_orig.modules[1].kW -- desired 1st layer feature size (supposed square!)
is2 = net_orig.modules[4].kW -- desired 2nd layer feature size (supposed square!)
dW1 = net_orig.modules[3].dW
dH1 = net_orig.modules[3].dH
dW2 = net_orig.modules[6].dW
dH2 = net_orig.modules[6].dH


kernels1 = kernels1[{{1,nk1}}]
kernels2 = kernels2[{{1,nk2}}] -- just take the 1st 'nk' kernels and use these

osz1 = torch.sqrt(kernels1:size(2)) -- (supposed square!)
osz2 = torch.sqrt(kernels2:size(2)) -- (supposed square!)

kernels1 = kernels1:reshape(nk1, osz1, osz1)
kernels2 = kernels2:reshape(nk2, osz2, osz2)

weight1 = torch.Tensor(nk1,is1,is1)
weight2 = torch.Tensor(nk2,is2,is2)

for i=1, nk1 do 
   weight1[i] = image.scale(kernels1[i], is1, is1) -- if not 7x7 or as required size!
end
for i=1, nk2 do 
   weight2[i] = image.scale(kernels2[i], is2, is2)
end

if opt.archn == 1 then
   net_new = nn.Sequential()
   net_new.modules[1] = net_orig.modules[1]:clone()
   net_new:add(nn.Tanh():float())
   net_new:add(nn.SpatialSubSampling(nk1, dW1, dW1, dH1, dH1):float())
   net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel):float())
   net_new.modules[5] = net_orig.modules[4]:clone()
   net_new:add(nn.Tanh():float())
   net_new:add(nn.SpatialSubSampling(nk2, dW2, dW2, dH2, dH2):float())
   net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel):float())
   net_new.modules[9] = net_orig.modules[7]:clone()
   
   -- initialize 1st layer parameters to learned filters:
   for i=1, nk1 do   
      net_new.modules[1].weight[i] = weight1[i]:type('torch.FloatTensor')
   end
   net_new.modules[1].bias = net_new.modules[1].bias *0
   -- initialize 2nd layer parameters to learned filters:
   for i=1, nk2 do   
      net_new.modules[5].weight[i] = weight2[i]:type('torch.FloatTensor')
   end
   net_new.modules[5].bias = net_orig.modules[4].bias *0

else
   net_new = net_orig:clone()

   -- initialize 1st layer parameters to learned filters:
   for i=1, nk1 do   
      net_new.modules[1].weight[i] = weight1[i]:type('torch.FloatTensor')
   end
   net_new.modules[1].bias = net_new.modules[1].bias *0
   -- initialize 2nd layer parameters to learned filters:
   for i=1, nk2 do   
      net_new.modules[4].weight[i] = weight2[i]:type('torch.FloatTensor')
   end
   net_new.modules[4].bias = net_orig.modules[4].bias *0
end

-- test net:
inp = torch.Tensor(3,46,46)
out = net_new:forward(inp:float())

-- save final product network:
torch.save('encoder_new.net', net_new)



