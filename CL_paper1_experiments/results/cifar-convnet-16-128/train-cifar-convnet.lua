----------------------------------------------------------------------
-- test standard convnet on CIFAR10 dataset - full convnet test 
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-visualize', true, 'display stuff')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', true, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:text()
params = cmd:parse(arg or {})
opt = cmd:parse(arg or {}) -- pass parameters to training files:

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)
torch.setdefaulttensortype('torch.DoubleTensor')


----------------------------------------------------------------------
print '==> loading dataset'
dofile '1_data_cifar.lua'


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
nstates = {16,128,128} -- OL: 16,128,128
fanin = {1,4}
filtsize = 5 -- OL has 7x7
poolsize = 2
normkernel = image.gaussian1D(7)


----------------------------------------------------------------------
print "==> processing dataset with standard 2 layer convnet"

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
omsize = ((32-(filtsize-1))/2 - (filtsize-1))/2
--omsize = (32-(filtsize-1))/2
model:add(nn.Reshape(nstates[2]*omsize*omsize))
model:add(nn.Linear(nstates[2]*omsize*omsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))


print "==> test network output:"
print(model:forward(trainData.data[1]:double()))

dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end





