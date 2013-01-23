----------------------------------------------------------------------
-- create a Online Learner neural net based on clustering learning filters
-- E. Culurciello, August 22nd 2012
----------------------------------------------------------------------

require 'nnx'
require 'image'
require 'eex'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Generates a new online-learner network')
cmd:text()
cmd:text('Options')
cmd:option('-nettype', 'random2l', 'Net type: "SAD", "randomf" (forest)... more')
-- comment: 1 = new architecture with SpatialSAD. Does not produce good nets for OL yet.
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:

torch.setdefaulttensortype('torch.FloatTensor')

net_orig = torch.load('encoder_orig.net') -- this is the original OL network renamed - a copy has to be in this folder!

normkernel = image.gaussian1D(7)


if not(layer1) then layer1 = torch.load('ol-layer1.net') end
if not(layer2) then layer2 = torch.load('ol-layer2.net') end
if not(layer3) then layer3 = torch.load('ol-layer3.net') end



if opt.nettype == 'SAD' then
   
   -- 3 layers SAD net
   net_new = nn.Sequential()
   for i=1,layer1:size() do net_new:add(layer1.modules[i]) end
   for i=1,layer2:size() do net_new:add(layer2.modules[i]) end
   for i=1,layer3:size() do net_new:add(layer3.modules[i]) end  
   
   -- init layer for OL inferring sizes later
   net_new.modules[1].gradInput = torch.zeros(3,46,46)
   


elseif opt.nettype == 'SAD2l' then
   
   -- 2 layers SAD net only
   net_new = nn.Sequential()
   for i=1,layer1:size() do net_new:add(layer1.modules[i]) end
   for i=1,layer2:size() do net_new:add(layer2.modules[i]) end
   
   -- init layer for OL inferring sizes later
   net_new.modules[1].gradInput = torch.zeros(3,46,46)



-- random [1st], 2nd layer network - currently gives the best results for OL nets   
elseif opt.nettype == 'random2l' then

   rnd1 = 0  -- random 1st layer?

   if not(rnd1) then kernels1 = torch.load('ol-1l-32-weights.net') end
   poolsize = 2
   if not(rnd1) then nk1 = kernels1:size(1) else nk1=32 end
   nk2 = 32
   fanin1 = 1
   fanin2 = 4
   is1 = 7
   is2 = 7  
  
   net_new = nn.Sequential()
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin1), is1, is1))
   net_new:add(nn.Tanh())
   --net_new:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)) -- this also works fine
   --net_new:add(nn.SpatialSubSampling(nk1, poolsize, poolsize, poolsize, poolsize)) -- this also works well
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   --net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel)) -- without this also works fine
   -- 2nd layer:
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
   net_new:add(nn.Tanh())
   --net_new:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
   --net_new:add(nn.SpatialSubSampling(nk2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   --net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   
   -- init layer for OL inferring sizes later
   net_new.modules[1].gradInput = torch.zeros(3,46,46)
   if not(rnd1) then net_new.modules[1].weight = kernels1 end
   if not(rnd1) then net_new.modules[1].bias = net_new.modules[1].bias*0 end



-- volumetic convolution: process multiple frames
elseif opt.nettype == 'vol2l' then

   rnd1 = 1  -- random 1st layer?
      
   if not(rnd1) then kernels1 = torch.load('ol-1l-32-weights.net') end
   poolsize = 2
   if not(rnd1) then nk1 = kernels1:size(1) else nk1=32 end
   nk2 = 32
   fanin1 = 1
   fanin2 = 4
   is1 = 7
   is2 = 7
   
   net_new = nn.Sequential()
   --(VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
   net_new:add(nn.VolumetricConvolution(3, nk1, 2, is1, is1))
   net_new:add(nn.Sum(2))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   --net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel)) -- without this also works fine
   -- 2nd layer:
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   --net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   
   -- init layer for OL inferring sizes later
   net_new.modules[1].gradInput = torch.zeros(3,46,46)
   if not(rnd1) then net_new.modules[1].weight = kernels1 end
   if not(rnd1) then net_new.modules[1].bias = net_new.modules[1].bias*0 end
   
   -- Volumetric Convolutions tests:
   inp = torch.Tensor(3,2,64,64) -- 3 planes, 2 frames of 64x64 each
   out3 = net_new:forward(inp:float()) -- should be 32 x 29 x 29

   

elseif opt.nettype == 'rforest' then   
   
   rnd1 = 1  -- random 1st layer?
      
   if not(rnd1) then kernels1 = torch.load('ol-1l-32-weights.net') end
   poolsize = 2
   if not(rnd1) then nk1 = kernels1:size(1) else nk1=32 end
   nk2 = 32
   fanin1 = 1
   fanin2 = 4
   is1 = 7
   is2 = 7  
  
   -- 1st layer:
   net_new = nn.Sequential()
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin1), is1, is1))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   --net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel)) -- without this also works fine
   
   -- 2nd layer: create up to lnmax random forest of parallel layers:
   lnmax = 10 
   net_newc=nn.Concat(1)
   for ln = 1, lnmax do
      net_new2 = nn.Sequential()
      net_new2:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
      net_new2:add(nn.Tanh())
      net_new2:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize))
      --net_new2:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
      net_newc:add(net_new2)
   end
   net_new:add(net_newc)
   --net_new:add(nn.Sum(1))
   
   net_new.modules[1].gradInput = torch.zeros(3,46,46) -- init layer for OL inferring sizes later
   if not(rnd1) then net_new.modules[1].weight = kernels1 end
   if not(rnd1) then net_new.modules[1].bias = net_new.modules[1].bias*0 end


elseif opt.nettype == 'SADrforest' then   
   net_new = nn.Sequential()
   net_new:add(layer1)
   
   -- create up to lnmax random forest of parallel layers:
   lnmax = 10 
   net_newc=nn.Concat(1)
   for ln = 1, lnmax do
      net_new2 = nn.Sequential()
      net_new2:add(layer2)
      net_newc:add(net_new2)
   end
   net_new:add(net_newc)
   --net_new:add(nn.Sum(1))
   
   net_new.modules[1].gradInput = torch.zeros(3,46,46) -- init layer for OL inferring sizes later
      
   
   
   
else print('error: no network type specified')
   
end

-- test net:
inp = torch.Tensor(3, 64,64)
out2 = net_new:forward(inp:float())

inp = torch.Tensor(3,46,46)
out1 = net_new:forward(inp:float())

-- save final product network:
torch.save('../online-learner/CL.net', net_new)
--if opt.archn == 1 then torch.save('../online-learner/CL3.net', net_new) else torch.save('../online-learner/CL2.net', net_new) end

