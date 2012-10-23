
-- test one trained net in another dataset:

import 'torch'
require 'image'
require 'unsup'

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
cmd:option('-nkernels', 512, 'number of kernels to learn')
cmd:option('-niter', 50, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 1000*100, 'nb of random training samples')
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

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
testLogger = optim.Logger('xtest.log')


----------------------------------------------------------------------
-- train, test classifier on SVHN:

-- load trained models SVHN:
model_svhn = torch.load('results/convnet-svhn/model.net')
convnet = model_svhn:clone()
convnet.modules[8]=nil

-- loading and processing CIFAR dataset:
dofile '1_data_cifar.lua'

-- convnet output size:
y1 = convnet:forward(trainData.data[1]:double())
convnetoutsize = y1:size(1)*y1:size(2)*y1:size(3)--256*5*5

print '==> processing dataset with CIFAR:'
trainData2 = {
   data = torch.Tensor(trsize, convnetoutsize),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, convnetoutsize),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = convnet:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size() do
   testData2.data[t] = convnet:forward(testData.data[t]:double())
   xlua.progress(t, testData:size())
end

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2

print "==> creating 2-layer network classifier"
opt.model = '2mlp-classifier'
nk = 256
l1netoutsize = 5
dofile '2_model.lua' 

print "==> test network output:"
print(model:forward(trainData.data[1]:clone()))

dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end

--------------------------------------------------------------
torch.load('c') -- break function
--------------------------------------------------------------


----------------------------------------------------------------------
-- train, test classifier on SVHN:

-- load trained models CIFAR:
model_cifar = torch.load('results/convnet-cifar/model.net')
convnet = model_cifar:clone()
convnet.modules[8]=nil

-- loading and processing SVHN dataset:
dofile '1_data_svhn.lua'

-- convnet output size:
y1 = convnet:forward(trainData.data[1]:double())
convnetoutsize = y1:size(1)*y1:size(2)*y1:size(3)--256*5*5

print '==> processing dataset with SVHN:'
trainData2 = {
   data = torch.Tensor(trsize, convnetoutsize),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, convnetoutsize),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = convnet:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size() do
   testData2.data[t] = convnet:forward(testData.data[t]:double())
   xlua.progress(t, testData:size())
end

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2

print "==> creating 2-layer network classifier"
opt.model = '2mlp-classifier'
nk = 256
l1netoutsize = 5
dofile '2_model.lua' 

print "==> test network output:"
print(model:forward(trainData.data[1]:clone()))

dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end





--------------------------------------------------------------
torch.load('c') -- break function
--------------------------------------------------------------


-- load trained models:
model_cifar = torch.load('results/convnet-cifar/model.net')
model_svhn = torch.load('results/convnet-svhn/model.net')

--------------------------------------------------------------
-- test net trained on cifar on svhn dataset: substitute convolutional layers
model = model_cifar
-- replace output classifier with svhn net classifier
model.modules[10] = model_svhn.modules[10]
model.modules[12] = model_svhn.modules[12]
dofile '1_data_svhn.lua'
test()

--------------------------------------------------------------
-- test net trained on cifar on svhn dataset:
model = model_cifar
dofile '5_test.lua'
test()

--------------------------------------------------------------
-- test net trained on svhn on svhn dataset:
model = model_svhn
dofile '5_test.lua'
test()


--------------------------------------------------------------
-- test net trained on svhn on cifar dataset: substitute convolutional layers
model = model_svhn
-- replace output classifier with svhn net classifier
model.modules[10] = model_cifar.modules[10]
model.modules[12] = model_cifar.modules[12]
dofile '1_data_cifar.lua'
dofile '5_test.lua'
test()

--------------------------------------------------------------
-- test net trained on svhn on cifar dataset:
model = model_svhn
dofile '5_test.lua'
test()

--------------------------------------------------------------
-- test net trained on cifar on cifar dataset:
model = model_cifar
dofile '5_test.lua'
test()
