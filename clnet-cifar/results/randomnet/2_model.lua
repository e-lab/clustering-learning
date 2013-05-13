----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models for Clustering Learning research
--
-- E. Culurciello modified original from Clement Farabet
-- August 2012
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides all sorts of trainable modules/layers
require 'eex'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {16,256,128}
fanin = {1,4}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(3)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == '1st-layer' then

   o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
   cvstepsize = 1
   poolsize = 2
   l1netoutsize = o1size/poolsize/cvstepsize
   
   model = nn.Sequential()
   model:add(nn.SpatialConvolution(3, nk1, is, is, cvstepsize, cvstepsize))
   --model:add(nn.HardShrink(0.5))
   model:add(nn.Tanh())
   --model:add(nn.SpatialSubSampling(nk1, poolsize, poolsize, poolsize, poolsize))
   --model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
   model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))


elseif opt.model == '2nd-layer' then

   o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
   cvstepsize = 1
   poolsize = 2
   l1netoutsize = o1size/poolsize/cvstepsize
   
   model = nn.Sequential()
   model:add(nn.SpatialConvolution(nk1, nk2, is, is, cvstepsize, cvstepsize))
   --model:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, 8), is, is))
   --model:add(nn.HardShrink(0.5))
   model:add(nn.Tanh())
   --model:add(nn.SpatialSubSampling(nk2, poolsize, poolsize, poolsize, poolsize))
   --model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
   model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
   
elseif opt.model == '1st-layer-dist' then   
   
   o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
   poolsize = 2
   l1netoutsize = o1size/poolsize

   model = nn.Sequential()
   model:add(nn.SpatialSAD(3, nk1, is, is))
   model:add(nn.Reshape(nk1*o1size*o1size))
   model:add(nn.Mul(nk1*o1size*o1size))
   model:add(nn.Reshape(nk1,o1size,o1size))

   model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))
   
   model:add(nn.Reshape(nk1*o1size*o1size))
   model:add(nn.Mul(nk1*o1size*o1size))
   model:add(nn.Reshape(nk1,o1size,o1size))

   model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
   model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

elseif opt.model == '2nd-layer-dist' then   
   
   o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
   poolsize = 2
   l1netoutsize = o1size/poolsize

   model = nn.Sequential()
   model:add(nn.SpatialSAD(nk1, nk2, is, is))
   model:add(nn.Reshape(nk2*o1size*o1size))
   model:add(nn.Mul(nk2*o1size*o1size))
   model:add(nn.Reshape(nk2,o1size,o1size))
 
   model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))
      
   model:add(nn.Reshape(nk2*o1size*o1size))
   model:add(nn.Mul(nk2*o1size*o1size))
   model:add(nn.Reshape(nk2,o1size,o1size))
   model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
   model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))


elseif opt.model == '2mlp-classifier' then
   
   nhiddens = 128
   outsize = 10 -- in CIFAR, SVHN datasets

   model = nn.Sequential()
   model:add(nn.Reshape(nk*l1netoutsize^2))
   model:add(nn.Linear(nk*l1netoutsize^2, nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,outsize))

   
elseif opt.model == '2mlp-cl-2layers' then

   nhiddens = 256
   outsize = 10 -- in CIFAR, SVHN datasets

   model = nn.Sequential()
   model:add(nn.Linear(nk2*l1netoutsize^2 + nk1*l1o^2, nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,outsize))


elseif opt.model == 'convnet' then

   -- a typical convolutional network, with locally-normalized hidden
   -- units, and L2-pooling

   -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
   -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
   -- the use of LP-pooling (with P=2) has a very positive impact on
   -- generalization. Normalization is not done exactly as proposed in
   -- the paper, and low-level (first layer) features are not fed to
   -- the classifier.

   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMap(nn.tables.random(nfeats, nstates[1], fanin[1]), filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1], nstates[2], fanin[2]), filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

   -- stage 3 : standard 2-layer neural network
   model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
   model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
   model:add(nn.Tanh())
   model:add(nn.Linear(nstates[3], noutputs))

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   if opt.model == 'convnet' then
      print '==> visualizing ConvNet filters'
      image.display{image=model:get(1).weight, padding=2, zoom=4, legend='filters @ layer 1'}
      image.display{image=model:get(5).weight, padding=2, zoom=4, nrow=32, legend='filters @ layer 2'}
   end
end
