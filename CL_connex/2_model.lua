----------------------------------------------------------------------
-- Author : Aysegul Dundar
-- This script demonstrates how to define a couple of different
-- models for Clustering Learning research
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nnx'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print '==> construct model'

if opt.model == '1st-layer' then
   
   model = nn.Sequential()
   if opt.whitening then 
      model:add(nn.SpatialConvolution(ivch, nk1, is1, is1, is1, is1))
   else 
      model:add(nn.SpatialConvolution(ivch, nk1, is1, is1))
   end
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))


elseif opt.model == '2nd-layer' then

   model = nn.Sequential()
   model:add(nn.SpatialConvolutionMap(cTable2, is2, is2))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))

elseif opt.model == '2mlp-classifier' then
   
   outsize = 10 -- in CIFAR, SVHN datasets

   model = nn.Sequential()
   model:add(nn.Linear(l1netoutsize+cdatasize, nhiddens))
   model:add(nn.Threshold())
   model:add(nn.Linear(nhiddens,outsize))

else

   error('unknown -model')

end


if verbose then print(model) end