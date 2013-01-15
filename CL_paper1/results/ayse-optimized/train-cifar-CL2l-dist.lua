----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 2st layer generation and test
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'
require 'eex'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 3, 'size of each input patches')
cmd:option('-nkernels', 128, 'number of kernels to learn')
cmd:option('-niter', 30, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 1000*1000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:option('-savedataset', false, 'save modified dataset')
cmd:option('-classify', true, 'run classification train/test')
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
opt = cmd:parse(arg or {}) -- pass parameters to training files:

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

is = opt.inputsize
nk = opt.nkernels


----------------------------------------------------------------------
print '==> loading pre-processed dataset with 1st layer clustering (test-cifar-1l-dist.lua)'

trsize = 20000
tesize = 2000

trainData = torch.load('trainData-cifar-CL1l-dist.t7')
testData = torch.load('testData-cifar-CL1l-dist.t7')

nk1=testData.data:size(2)

----------------------------------------------------------------------
print "==> preparing images"
-- remove offsets
trainData.data=trainData.data-torch.mean(trainData.data)
testData.data=testData.data-torch.mean(testData.data)


----------------------------------------------------------------------
--[[print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data = torch.Tensor(opt.nsamples,nk1*is*is)
for i = 1,opt.nsamples do
   img = math.random(1,trainData.data:size(1))
   img2 = trainData.data[img]
   z = math.random(1,trainData.data:size(2))
   x = math.random(1,trainData.data:size(3)-is+1)
   y = math.random(1,trainData.data:size(4)-is+1)
   randompatch = img2[{ {},{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data[i] = randompatch
end
]]
-- show a few patches:
if opt.visualize then
   --f256S = data[{{1,256}}]:reshape(256,is,is)
   --image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}
end

--if not paths.filep('cifar10-1l.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         --win = image.display{image=kernels:reshape(nk,is,is), padding=2, symmetric=true, 
         --zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='2nd layer filters'}
      end
   end                    
--[[   kernels = unsup.kmeans(data, nk, opt.initstd,opt.niter, opt.batchsize,cb,true)
   print('==> saving centroids to disk:')
   torch.save('cifar10-2l.t7', kernels)]]
print('loading kernels')
kernels=torch.load('cifar10-2l.t7')
--else
--   print '==> loading pre-trained k-means kernels'
--   kernels = torch.load('cifar10-2l.t7')
--end

for i=1,nk do
   -- normalize kernels to 0 mean and 1 std:
 --  kernels[i]:add(-kernels[i]:mean())
 --  kernels[i]:div(kernels[i]:std())

   -- clear nan kernels   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end

print '==> verify filters statistics'
print('filters max mean: ' .. kernels:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels:std(2):abs():max())

--kernels = torch.load('cifar10-2l-256.t7')
--kernels = kernels[1][{{1,nk}}] -- just take the 1st 'nk' kernels and use these


----------------------------------------------------------------------
print "==> loading and initializing 2nd layer CL model"

nk1=testData.data:size(2)
nk2=nk
opt.model = '2nd-layer-dist'
dofile '2_model.lua' 
l1net = model:clone()

-- initialize templates:
--l1net.modules[1]:templates(kernels:reshape(nk2, 1, is, is):expand(nk2,nk1,is,is))
l1net.modules[1]:templates(kernels)
l1net.modules[1].bias = l1net.modules[1].bias *0
l1net.modules[3].weight = torch.ones(1)*(-1)
l1net.modules[7].weight = torch.ones(1)*(1/is)*(1/is)*(1/3)


torch.save('l1net.net', l1net)

l1net:forward(trainData.data[1]:double())
--print(l1net.modules[1].output)
--print('DDD')

print('1', l1net.modules[1], l1net.modules[1].output:min(), l1net.modules[1].output:max())
print('2', l1net.modules[2], l1net.modules[2].output:min(), l1net.modules[3].output:max())
print('3', l1net.modules[3], l1net.modules[3].output:min(), l1net.modules[3].output:max())
print('4', l1net.modules[4], l1net.modules[4].output:min(), l1net.modules[4].output:max())
print('5', l1net.modules[5], l1net.modules[5].output:min(), l1net.modules[5].output:max())
print('6', l1net.modules[6], l1net.modules[6].output:min(), l1net.modules[6].output:max())
print('7', l1net.modules[7], l1net.modules[7].output:min(), l1net.modules[7].output:max())
print('8', l1net.modules[8], l1net.modules[8].output:min(), l1net.modules[8].output:max())
print('9', l1net.modules[9], l1net.modules[9].output:min(), l1net.modules[9].output:max())
print('10', l1net.modules[10], l1net.modules[10].output:min(), l1net.modules[10].output:max())
----------------------------------------------------------------------
print "==> processing dataset with CL network"

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
   trainData2.data[t] = l1net:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())
for t = 1,testData:size() do
   testData2.data[t] = l1net:forward(testData.data[t]:double())
   xlua.progress(t, testData:size())
end
--testData2.data = l1net:forward(testData.data:double())

trainData2.data = trainData2.data:reshape(trsize, nk, l1netoutsize, l1netoutsize)
testData2.data = testData2.data:reshape(tesize, nk, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2

-- show a few outputs:
if opt.visualize then
   f256S_y = trainData2.data[{ {1,256},1 }]
   image.display{image=f256S_y, nrow=16, nrow=16, padding=2, zoom=2, 
            legend='Output 2nd layer: first 256 examples, 1st feature'}
end

print '==> verify statistics'
channels = {'r','g','b'}
for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

--print(trainData2.data[210000]) --break function


----------------------------------------------------------------------
-- save datasets:
if opt.savedataset then
   trainData.data = trainData.data:float()  -- float to save space if needed
   testData.data = testData.data:float()
   torch.save('trainData-cifar-CL2l-dist.t7', trainData)
   torch.save('testData-cifar-CL2l-dist.t7', testData)
end


----------------------------------------------------------------------
-- classifier for train/test:
if opt.classify then
   ----------------------------------------------------------------------
   print "==> creating classifier"
   
   opt.model = '2mlp-classifier'
   dofile '2_model.lua' 
   
   print "==> test network output:"
   print(model:forward(trainData.data[1]:double()))
   
   dofile '3_loss.lua' 
   dofile '4_train.lua'
   dofile '5_test.lua'
   
   ----------------------------------------------------------------------
   print "==> training classifier"
   
   while true do
      train()
      test()
   end
   
end






