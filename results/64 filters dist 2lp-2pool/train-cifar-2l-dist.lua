----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 2st layer generation and test
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'
require 'SpatialSAD'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-images', 'images', 'directory full of images')
cmd:option('-maximages', 100, 'max nb of images')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches') -- 9x9 kernels wanted
cmd:option('-nkernels', 16, 'number of kernels to learn')
cmd:option('-niter', 50, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 1000*100*1, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
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

if not qt then
   opt.visualize = false
end

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

is = params.inputsize
nk = params.nkernels

----------------------------------------------------------------------
print '==> loading pre-processed dataset with 1st layer clustering (test-cifar-1l-dist.lua)'

trsize = 50000
tesize = 2000

trainData = torch.load('trainData-l1-64-d.t7')
testData = torch.load('testData-l1-64-d.t7')

----------------------------------------------------------------------
print "==> preparing images"

print '==> extracting patches'
data = torch.Tensor(params.nsamples,64*is*is)
for i = 1,params.nsamples do
   local img = math.random(1,trainData.data:size(1))
   local image = trainData.data[img]
   local x = math.random(1,trainData.data:size(3)-is+1)
   local y = math.random(1,trainData.data:size(4)-is+1)
   local randompatch = image[{ {},{y,y+is-1},{x,x+is-1} }]
   data[i] = randompatch
end

if not paths.filep('cifar10-2l-64.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk,3,is,is),
                          padding=2, symmetric=true, zoom=2, win=win,
                          nrow=math.floor(math.sqrt(nk)),
                          legend='K-Means Centroids'}
      end
   end                    
   kernels = unsup.kmeans(data, nk, params.initstd, params.niter, params.batchsize,cb,true)
   print('==> saving centroids to disk:')
   torch.save('cifar10-2l.t7', kernels)
else
   print '==> loading pre-trained k-means kernels'
   kernels = torch.load('cifar10-2l.t7')
end

-- there is a bug in unpus.kmeans: some kernels come out nan!!!
-- clear nan kernels
for i=1,nk do   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end

----------------------------------------------------------------------
print "==> processing dataset with k-means kernels + pooling"

normkernel = image.gaussian1D(7)
o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
poolsize = 4
cvstepsize = 1 -- convolution step size, not used in SAD
outsize = torch.max(trainData.labels)

l1net = nn.Sequential()
l1net:add(nn.SpatialSAD(3,nk,is,is))
l1net.modules[1]:templates(kernels:reshape(nk, 3, is, is))
l1net:add(nn.SpatialContrastiveNormalization(nk, normkernel, 1e-3))
l1net:add(nn.Tanh())
l1net:add(nn.SpatialLPPooling(nk,2,poolsize,poolsize,poolsize,poolsize))

-- output size:
l1netoutsize = o1size/poolsize/cvstepsize

-- processing dataset:
trainData2 = {
   data = torch.Tensor(trsize, nk*(l1netoutsize)^2),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, nk*(l1netoutsize)^2),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = l1net:forward(trainData.data[t])
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size() do
   testData2.data[t] = l1net:forward(testData.data[t])
   xlua.progress(t, testData:size())
end

trainData2.data = trainData2.data:reshape(trsize, nk, l1netoutsize, l1netoutsize)
testData2.data = testData2.data:reshape(tesize, nk, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2


----------------------------------------------------------------------
print "==> creating 2-layer network classifier"

nhiddens = outsize^2

model = nn.Sequential()
model:add(nn.Reshape(nk*(l1netoutsize)^2))
model:add(nn.Linear(nk*(l1netoutsize)^2,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,outsize))

print "==> test network output:"
print(model:forward(trainData.data[1]))

dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end






