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
cmd:option('-nkernels', 256, 'number of kernels to learn')
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

--[[
print '==> verify statistics'
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
testMean = testData.data:mean()
testStd = testData.data:std()
print('training data mean: ' .. trainMean)
print('training datastandard deviation: ' .. trainStd)
print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)]]



std = torch.std(trainData.data)
trainData.data=(trainData.data-torch.mean(trainData.data))/std
testData.data=(testData.data-torch.mean(testData.data))/std


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



print '==> running k-means'
function cb (kernels)
   if opt.visualize then
         --win = image.display{image=kernels[1]:reshape(nk2,is,is), padding=2, symmetric=true, 
         --zoom=2, win=win, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters, 1plane'}
   end
end                    
kernels, kcounts = unsup.kmeans(data, nk2, opt.initstd, opt.niter, opt.batchsize,cb,true)
print('==> saving centroids to disk:')
torch.save('svhn-CL2l.t7', kernels)


--kernels = torch.load('svhn-CL2l.t7')

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


--print '==> verify filters statistics'
--print('filters max mean: ' .. kernels:mean(2):abs():max())
--print('filters max standard deviation: ' .. kernels:std(2):abs():max())
--kernels, kcounts= torch.load('svhn-CL2l.t7')
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
opt.model = '2nd-layer-dist'
dofile '2_model.lua' 
l1net = model:clone()

-- initialize templates:
--l1net.modules[1]:templates(kernels:reshape(nk2, 1, is, is):expand(nk2,nk1,is,is))
l1net.modules[1]:templates(kernels)
l1net.modules[1].bias = l1net.modules[1].bias *0
l1net.modules[3].weight = torch.ones(1)*(-1)
l1net.modules[7].weight = torch.ones(1)*(1/is)*(1/is)*(1/12)


l1net:forward(trainData.data[1]:double())


print('1', l1net.modules[1], l1net.modules[1].output:min(), l1net.modules[1].output:max())
print('2', l1net.modules[2], l1net.modules[2].output:min(), l1net.modules[2].output:max())
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




