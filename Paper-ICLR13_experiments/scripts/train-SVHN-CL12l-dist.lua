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
nk = nk2


----------------------------------------------------------------------
print '==> loading pre-processed dataset with 1st layer clustering (test-SVHN-CL1l.lua)'

trainData1 = torch.load('trainData-svhn-CL1l-dist.t7')
testData1 = torch.load('testData-svhn-CL1l-dist.t7')
trainData = trainData1 -- temporary pointers
testData = testData1

nk1=testData.data:size(2)
-- ATTENTION HERE: MIGHT HAVE BEEN REDUCED TO SPEED UP TEST!!!! check model file
trsize = trainData.data:size(1)
tesize = testData.data:size(1)


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

vs = testData.data:size(3)
if opt.visualize then
   image.display{image=trainData.data[{{1,256},{1}}]:reshape(256,vs,vs),padding=2, nrow=16, 
         symmetric=true, zoom=2, legend='Some normalized training samples'}
   image.display{image=testData.data[{{1,256},{1}}]:reshape(256,vs,vs),padding=2, nrow=16, 
         symmetric=true, zoom=2, legend='Some normalized test samples'}
end

----------------------------------------------------------------------
--print "==> preparing images"
-- remove offsets
--trainData.data=trainData.data-torch.mean(trainData.data)
--testData.data=testData.data-torch.mean(testData.data)


print '==> extracting patches'
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

-- show a few patches:
if opt.visualize then
   --f256S = data[{{1,256}}]:reshape(256,is,is)
   --image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}
end

print '==> running k-means'
function cb (kernels)
   if opt.visualize then
         --win = image.display{image=kernels[1]:reshape(nk2,is,is), padding=2, symmetric=true, 
         --zoom=2, win=win, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters, 1plane'}
   end
end                    
kernels, kcounts = unsup.kmeans(data, nk2, opt.initstd, opt.niter, opt.batchsize,cb,true)
print('==> saving centroids to disk:')
torch.save('svhn-CL2l.t7', {kernels, kcounts})

for i=1,nk2 do
   -- normalize kernels to 0 mean and 1 std:
   kernels[i]:add(-kernels[i]:mean())
   kernels[i]:div(kernels[i]:std())

   -- clear nan kernels   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end


print '==> verify filters statistics'
print('filters max mean: ' .. kernels:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels:std(2):abs():max())

--kernels = torch.load('svhn-CL2l-256.t7')
--kernels = kernels[1][{{1,nk2}}] -- just take the 1st 'nk' kernels and use these


----------------------------------------------------------------------
print "==> using patches as filters"

--kernels = torch.Tensor(nk2,is,is)
--kernels[{{1,64}}] = torch.load('svhn-CL1l.t7') -- resuse 1st layer kernels
--kernels[{{65,nk2}}] = data[{{65,nk2}}]
-- normalize kernels to 0 mean and 1 std:
--for i=1,nk2 do
--   kernels[i]:add(-kernels[i]:mean())
--   kernels[i]:div(kernels[i]:std())
--end
--image.display{image=kernels, padding=2, zoom=4, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'}


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
l1net.modules[4].weight = torch.ones(1)*(1/is*1/2) -- the value 2/is is empirical, looking at net output...


----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData2 = {
   data = torch.Tensor(trsize, nk*(l1netoutsize)^2),
   labels = trainData1.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, nk*(l1netoutsize)^2),
   labels = testData1.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = l1net:forward(trainData1.data[t]:double())
   xlua.progress(t, trainData1:size())
end
--trainData2.data = l1net:forward(trainData1.data:double())
for t = 1,testData:size() do
   testData2.data[t] = l1net:forward(testData1.data[t]:double())
   xlua.progress(t, testData1:size())
end
--testData2.data = l1net:forward(testData1.data:double())


-- show a few outputs:
--if opt.visualize then
--   f256S_y = trainData2.data[{ {1,256},1 }]
--   image.display{image=f256S_y, nrow=16, nrow=16, padding=2, zoom=2, 
--            legend='Output 2nd layer: first 256 examples, 1st feature'}
--end

print '==> verify statistics'
channels = {'r','g','b'}
for i,channel in ipairs(channels) do
   trainMean = trainData2.data[{ {},i }]:mean()
   trainStd = trainData2.data[{ {},i }]:std()

   testMean = testData2.data[{ {},i }]:mean()
   testStd = testData2.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end


----------------------------------------------------------------------
-- save datasets:
if opt.savedataset then
   trainData2.data = trainData2.data:float() -- float to save space if needed
   testData2.data = testData2.data:float()
   torch.save('trainData-SVHN-CL2l-dist.t7', trainData2)
   torch.save('testData-SVHN-CL2l-dist.t7', testData2)
end

----------------------------------------------------------------------
print "==> now we feed both layers outputs to a classifier"


-- create final dataset to be used by classifier:
l1o =trainData1.data:size(3)
trainData = {
   data = torch.Tensor(trsize, nk2*l1netoutsize^2 + nk1*l1o^2),
   labels = trainData1.labels:clone(),
   size = function() return trsize end
}
testData = {
   data = torch.Tensor(tesize, nk2*l1netoutsize^2 + nk1*l1o^2),
   labels = testData1.labels:clone(),
   size = function() return tesize end
}

-- concatenate data:
trainData.data[{{},{1, nk1*l1o^2}}] = trainData1.data
testData.data[{{},{1, nk1*l1o^2}}] = testData1.data
trainData.data[{{},{nk1*l1o^2+1, nk2*l1netoutsize^2 + nk1*l1o^2}}] = trainData2.data
testData.data[{{},{nk1*l1o^2+1, nk2*l1netoutsize^2 + nk1*l1o^2}}] = testData2.data


----------------------------------------------------------------------
-- classifier for train/test:
if opt.classify then
   ----------------------------------------------------------------------
   print "==> creating classifier"
   
   opt.model = '2mlp-cl-2layers'
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


