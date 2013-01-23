----------------------------------------------------------------------
-- create a Online Learner neural net based on clustering learning filters
-- E. Culurciello, August 22nd 2012
----------------------------------------------------------------------

require 'nnx'
require 'image'
require 'eex'
require 'MulAnySize'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Generates a new online-learner network')
cmd:text()
cmd:text('Options')
cmd:option('-archn', 0, 'create new architecture or leave old one intact?')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:

torch.setdefaulttensortype('torch.FloatTensor')

net_orig = torch.load('encoder_orig.net') -- this is the original OL network renamed - a copy has to be in this folder!
--image.display{image=net_orig.modules[1].weight, zoom=4, padding=2, nrow =4}  -- to visualize filters
--image.display{image=net_orig.modules[4].weight[{{1,128}}], zoom=4, padding=2, nrow =16}
--image.display{image=net_orig.modules[7].weight[{{1,128}}], zoom=4, padding=2, nrow =16}


kernels1 = torch.load('berkeley56x56-1l.t7')
kernels2 = torch.load('berkeley56x56-2l.t7')
kernels3 = torch.load('berkeley56x56-3l.t7')
kernels2w = torch.load('berkeley56x56-2l-w.t7')
kernels3w = torch.load('berkeley56x56-3l-w.t7')

--image.display{image=kernels1:reshape(nk1,is1,is1), padding=2, symmetric=true, 
--         zoom=2, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
--         
--image.display{image=kernels2:reshape(nk2,is2,is2), padding=2, symmetric=true, 
--         zoom=2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'}

-- original OL is 16,128,128 ouput planes/features
nk1 = net_orig.modules[1].nOutputPlane -- desired number of 1st layer features 
nk2 = net_orig.modules[4].nOutputPlane -- desired number of 2nd layer features
nk3 = net_orig.modules[7].nOutputPlane -- desired number of 3rd layer features

nk2w = net_orig.modules[4].weight:size(1) -- desired number of 2nd layer features
nk3w = net_orig.modules[7].weight:size(1) -- desired number of 3rd layer features

is1 = net_orig.modules[1].kW -- desired 1st layer feature size (supposed square!)
is2 = net_orig.modules[4].kW -- desired 2nd layer feature size (supposed square!)
is3 = net_orig.modules[7].kW -- desired 3rd layer feature size (supposed square!)

dW1 = net_orig.modules[3].dW
dH1 = net_orig.modules[3].dH
dW2 = net_orig.modules[6].dW
dH2 = net_orig.modules[6].dH


kernels1 = kernels1[{{1,nk1}}]
kernels2 = kernels2[{{1,nk2}}]:reshape(128,16,49) -- just take the 1st 'nk' kernels and use these
kernels3 = kernels3[{{1,nk3}}]

osz1 = torch.sqrt(kernels1:size(2)) -- (supposed square!)
osz2 = torch.sqrt(kernels2:size(3)) -- (supposed square!)
osz3 = kernels3:size(2)

-- 2nd layer filters are 128 filters with 16 planes, so need to extract some random ones:
kernels2f = torch.Tensor(nk2,is2*is2)
for i = 1,nk2 do
   x = math.random(1,nk1)
   y = math.random(1,nk2)
   kernels2f[i] = kernels2[{{y},{x},{},{}}]:reshape(7*7)
end


kernels1 = kernels1:reshape(nk1, osz1, osz1)
kernels2 = kernels2f:reshape(nk2, osz2, osz2)
kernels3 = kernels3:reshape(nk3, osz3, osz3)

--weight1 = torch.Tensor(nk1,is1,is1)
--weight2 = torch.Tensor(nk2,is2,is2)
--weight3 = torch.Tensor(nk3,is3,is3)
--
--if osz1 ~= 7 then
--   for i=1, nk1 do 
--      weight1[i] = image.scale(kernels1[i], is1, is1) -- if not 7x7 or as required size!
--   end
--end
--if osz2 ~= 7 then
--   for i=1, nk2 do 
--      weight2[i] = image.scale(kernels2[i], is2, is2)
--   end
--end
--if osz3 ~= 7 then
--   for i=1, nk3 do 
--      weight3[i] = image.scale(kernels3[i], is3, is3)
--   end
--end




if opt.archn == 1 then

   poolsize=2
   normkernel = image.gaussian1D(7)
   
   -- 1st layer:
   net_new = nn.Sequential()
   net_new:add(nn.SpatialSAD(3, nk1, is1, is1))
   net_new:add(nn.SpatialContrastiveNormalization(nk1, normkernel))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))
   -- 2nd layer:
   net_new:add(nn.SpatialSAD(nk1, nk2, is2, is2))
   net_new:add(nn.SpatialContrastiveNormalization(nk2, normkernel))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   --3rd layer:
   net_new:add(nn.SpatialSAD(nk2, nk3, is3, is3))
   --net_new:add(nn.SpatialContrastiveNormalization(nk3, normkernel))
   net_new:add(nn.MulAnySize(1))
   
   -- initialize 1st layer parameters to learned filters:
   net_new.modules[1]:templates(kernels1:reshape(nk1, 1, is1, is1):expand(nk1,3,is1,is1):float())
   net_new.modules[1].bias = net_orig.modules[1].bias*0 
   -- initialize 2nd layer parameters to learned filters: 
   net_new.modules[6]:templates(kernels2:reshape(nk2, 1, is2, is2):expand(nk2,nk1,is2,is2):float())
   net_new.modules[6].bias = net_orig.modules[4].bias*0
   -- initialize 3rd layer parameters to learned filters:
   net_new.modules[11]:templates(kernels3:reshape(nk3, 1, is3, is3):expand(nk3,nk2,is3,is3):float())
   net_new.modules[11].bias = net_orig.modules[7].bias*0
   net_new.modules[12].weight = torch.ones(1)*(1/is3)*(1/is3)*(1/2)

else
   net_new = net_orig:clone()

   -- initialize 1st layer parameters to learned filters:
   net_new.modules[1].weight = kernels1:float()
   net_new.modules[1].bias = net_orig.modules[1].bias*0
   -- initialize 2nd layer parameters to learned filters:
   net_new.modules[4].weight = kernels2w:float()
   net_new.modules[4].bias = net_orig.modules[4].bias*0
    -- initialize 3rd layer parameters to learned filters: 
   net_new.modules[7].weight = kernels3w:float()
   net_new.modules[7].bias = net_orig.modules[7].bias*0
end

-- test net:
inp = torch.Tensor(3,46,46)
out1 = net_new:forward(inp:float())

inp = torch.Tensor(3,64,64)
out2 = net_new:forward(inp:float())

-- save final product network:
torch.save('encoder_new.net', net_new)
torch.save('encoder.net', net_new)


