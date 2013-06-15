----------------------------------------------------------------------
-- Author : Aysegul Dundar
-- This script demonstrates how to define a couple of different
-- models for Clustering Learning research
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nnx'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- define network to train
--
print('<trainer> creating new network')

nk1           =  32          -- nb of features
is1,is2       =  5,5  	     -- size of kernels
ss1,ss2, ss3  =  2,2,4 		 -- size of subsamplers (strides)
fanin 	     =  2 		     -- createCoCnxTable creates also 2*fanin connections
feat_group 	  =  32 		 -- features per group (32=best in CIFAR nk1=32, fanin=2)
nhiddens      =  256 		 -- nb of hidden features for top perceptron (0=linear classifier)

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