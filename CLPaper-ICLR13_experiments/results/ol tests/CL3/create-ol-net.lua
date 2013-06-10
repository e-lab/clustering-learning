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
cmd:option('-nettype', 'random2l', 'Net type: "SAD", "orig", "randomf" (forest)')
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

-- random 2nd layer   
elseif opt.nettype == 'random2l' then

   --kernels1 = torch.load('ol-1l-32-weights.net')
   poolsize = 2
   nk1 = 64--kernels1:size(1)
   nk2 = 64
   fanin1 = 1
   fanin2 = 4
   is1 = 7
   is2 = 7  
  
   net_new = nn.Sequential()
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin1), is1, is1))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))
   -- 2nd layer:
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   
   -- init layer for OL inferring sizes later
   net_new.modules[1].gradInput = torch.zeros(3,46,46)
   --net_new.modules[1].weight = kernels1
   --net_new.modules[1].bias = net_new.modules[1].bias*0
   

elseif opt.nettype == 'orig' then

   layer1 = torch.load('ol-orig-layer1.net') -- load network created with berkeley sized as original OL 
   layer2 = torch.load('ol-orig-layer2.net')
   layer3 = torch.load('ol-orig-layer3.net')
   
   nk1 = net_orig.modules[1].nOutputPlane -- desired number of 1st layer features 
   nk2 = net_orig.modules[4].nOutputPlane -- desired number of 2nd layer features   
   nk3 = net_orig.modules[7].nOutputPlane -- desired number of 3rd layer features
   
   is1 = net_orig.modules[1].kW -- desired 1st layer feature size (supposed square!)
   is2 = net_orig.modules[4].kW -- desired 2nd layer feature size (supposed square!)
   is3 = net_orig.modules[7].kW -- desired 3rd layer feature size (supposed square!)
   
   poolsize = 2
   fanin1 = net_orig.modules[1].weight:size(1)/net_orig.modules[1].nOutputPlane
   fanin2 = net_orig.modules[4].weight:size(1)/net_orig.modules[4].nOutputPlane
   fanin3 = net_orig.modules[7].weight:size(1)/net_orig.modules[7].nOutputPlane
   
   -- layer 1:
   net_new = nn.Sequential()
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin1), is1, is1))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))
   -- layer 2:
   net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin2), is2, is2))
   net_new:add(nn.Tanh())
   net_new:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   net_new:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   --net_new:add(nn.SpatialConvolutionMap(nn.tables.random(nk2, nk3, fanin3), is3, is3))
   
   -- initialize 1st layer parameters to learned filters:
   net_new.modules[1].gradInput = torch.zeros(3,46,46) -- init layer for OL inferring sizes later
   net_new.modules[1].weight = layer1.modules[1].weight
   net_new.modules[1].bias = layer1.modules[1].bias
   -- no need to copy conn table of 1st layer, since it is trivial in original OL
   -- initialize 2nd layer parameters to learned filters:
   net_new.modules[5].weight = layer2.modules[1].weight
   net_new.modules[5].bias = layer2.modules[1].bias
   net_new.modules[5].connTable = layer2.modules[1].connTable
   -- initialize 3rd layer parameters to learned filters: 
   --net_new.modules[9].weight = layer3.modules[1].weight
   --net_new.modules[9].bias = layer3.modules[1].bias
   --net_new.modules[9].connTable = layer3.modules[1].connTable

   
elseif opt.nettype == 'randomf' then   
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
inp = torch.Tensor(3,64,64)
out2 = net_new:forward(inp:float())

inp = torch.Tensor(3,46,46)
out1 = net_new:forward(inp:float())

-- save final product network:
torch.save('../online-learner/CL.net', net_new)
--if opt.archn == 1 then torch.save('../online-learner/CL3.net', net_new) else torch.save('../online-learner/CL2.net', net_new) end

