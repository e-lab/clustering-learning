----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 2st layer generation and test
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 5, 'size of each input patches') -- 9x9 kernels wanted
cmd:option('-nkernels', 256, 'number of kernels to learn')
cmd:option('-niter', 1, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 100*1000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
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

----------------------------------------------------------------------
print '==> loading pre-processed dataset with 1st layer clustering (test-cifar-1l-dist.lua)'

trsize = 50000
tesize = 2000

trainData = torch.load('trainData-cifar-CL1l.t7')
testData = torch.load('testData-cifar-CL1l.t7')

print '==> verify statistics'
channels = {'y','u','v'}
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

if opt.visualize then
   image.display{image=trainData.data[{{1,256},{1}}]:reshape(256,12,12),padding=2, nrow=16, 
         symmetric=true, zoom=2, legend='Some normalized training samples'}
   image.display{image=testData.data[{{1,256},{1}}]:reshape(256,12,12),padding=2, nrow=16, 
         symmetric=true, zoom=2, legend='Some normalized test samples'}
end

nk1 = trainData.data:size(2)

----------------------------------------------------------------------
print "==> preparing images"
-- remove offsets
--trainData.data=trainData.data-torch.mean(trainData.data)
--testData.data=testData.data-torch.mean(testData.data)


print '==> extracting patches'
data = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   local img = math.random(1,trainData.data:size(1))
   local image = trainData.data[img]
   local z = math.random(1,trainData.data:size(2))
   local x = math.random(1,trainData.data:size(3)-is+1)
   local y = math.random(1,trainData.data:size(4)-is+1)
   local randompatch = image[{ {z},{y,y+is-1},{x,x+is-1} }]
   data[i] = randompatch
end

-- show a few patches:
if opt.visualize then
   f256S = data[{{1,256}}]:reshape(nk2,is,is)
   image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}
end

print '==> running k-means'
function cb (kernels)
   if opt.visualize then
         win = image.display{image=kernels:reshape(nk2,is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'}
   end
end                    
kernels, kcounts = unsup.kmeans(data, nk2, opt.initstd, opt.niter, opt.batchsize,cb,true)
print('==> saving centroids to disk:')
torch.save('cifar10-2l.t7', {kernels, kcounts})

for i=1,nk2 do
   -- there is a bug in unpus.kmeans: some kernels come out nan!!!
   -- clear nan kernels   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
   
   -- give gaussian shape if needed:
--   sigma=0.25
--   fil = image.gaussian(is, sigma)
--   kernels[i] = kernels[i]:cmul(fil)
   
-- normalize kernels to 0 mean and 1 std:
   kernels[i]:add(-kernels[i]:mean())
   kernels[i]:div(kernels[i]:std())
end

print '==> verify filters statistics'
print('filters max mean: ' .. kernels:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels:std(2):abs():max())

--kernels = torch.load('cifar10-2l-256.t7')
--kernels = kernels[1][{{1,nk2}}] -- just take the 1st 'nk' kernels and use these

----------------------------------------------------------------------
print "==> processing dataset with k-means kernels + pooling"

opt.model = '2nd-layer'
dofile '2_model.lua'
l1net = model:clone()

-- initialize templates:
l1net.modules[1].weight = kernels:reshape(nk2,1,is,is):expand(nk2,nk1,is,is)
l1net.modules[1].bias = l1net.modules[1].bias *0


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


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------


----------------------------------------------------------------------
--print "==> creating 1-layer network classifier"

print "==> creating 2-layer network classifier"
nk=nk2
opt.model = '2mlp-classifier'
dofile '2_model.lua' 

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


-- save datasets:
--trainData.data = trainData.data:float()
--testData.data = testData.data:float()
--torch.save('trainData-CL2l.t7', trainData)
--torch.save('testData-CL2l.t7', testData)





