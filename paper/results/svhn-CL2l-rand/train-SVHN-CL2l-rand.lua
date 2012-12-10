----------------------------------------------------------------------
-- Run k-means on SVHN dataset - learn 2nd layer filters
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
cmd:option('-inputsize', 5, 'size of each input patches') -- 9x9 kernels wanted
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

-- shortcuts:
is = opt.inputsize
nk2 = opt.nkernels
nk=nk2

----------------------------------------------------------------------
print '==> loading pre-processed dataset with 1st layer clustering (test-SVHN-CL1l.lua)'

trainData = torch.load('trainData-svhn-CL1l-dist.t7')
testData = torch.load('testData-svhn-CL1l-dist.t7')

-- ATTENTION HERE: MIGHT HAVE BEEN REDUCED TO SPEED UP TEST!!!! check model file
trsize = trainData.data:size(1)
tesize = testData.data:size(1)

nk1=testData.data:size(2)


print '==> verify statistics'
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
testMean = testData.data:mean()
testStd = testData.data:std()
print('training data mean: ' .. trainMean)
print('training datastandard deviation: ' .. trainStd)
print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)

----------------------------------------------------------------------
print "==> loading and initializing 2nd layer random convnet layer model"

nk1=testData.data:size(2)
nk2=nk
poolsize=2
normkernel = image.gaussian1D(7)
o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
l1netoutsize = o1size/poolsize

l1net = nn.Sequential()
l1net:add(nn.SpatialConvolution(nk1, nk2, is, is))
l1net:add(nn.Tanh())
l1net:add(nn.SpatialLPPooling(nk2,2,poolsize,poolsize,poolsize,poolsize))
l1net:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

-- initialized random, as it is!
--tests:
inp = torch.Tensor(100)
for t = 1,100 do
   l1net:forward(trainData.data[t]:double())
   inp[t] = l1net.modules[4].output:max()
end
print('MAX output after nn.Mul:', inp:mean())



----------------------------------------------------------------------
print "==> processing dataset with CL network"

-- processing dataset:
trainData2 = {
   data = torch.Tensor(trsize, nk2*(l1netoutsize)^2),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, nk2*(l1netoutsize)^2),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = l1net:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size() do
   testData2.data[t] = l1net:forward(testData.data[t]:double())
   xlua.progress(t, testData:size())
end

trainData2.data = trainData2.data:reshape(trsize, nk2, l1netoutsize, l1netoutsize)
testData2.data = testData2.data:reshape(tesize, nk2, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2


----------------------------------------------------------------------
-- save datasets:
if opt.savedataset then
   trainData.data = trainData.data:float()  -- float to save space if needed
   testData.data = testData.data:float()
   torch.save('trainData-SVHN-CL2l-dist.t7', trainData)
   torch.save('testData-SVHN-CL2l-dist.t7', testData)
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




