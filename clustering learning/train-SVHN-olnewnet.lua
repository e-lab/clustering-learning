----------------------------------------------------------------------
-- Test new online learner net (trained with clustering learning) on SVHN
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'
require 'SpatialSAD'
require 'image'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-maximages', 100, 'max nb of images')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
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


----------------------------------------------------------------------
print '==> loading dataset'
dofile '1_data_svhn.lua'


----------------------------------------------------------------------
print "==> resizing dataset to 46x46, required by online learner:"

olinsize = 46 -- online learner network image input size
-- resize all dataset images for the online learner net:
trainDataB = {
   data = torch.Tensor(trsize, 3*(olinsize)^2),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testDataB = {
   data = torch.Tensor(tesize, 3*(olinsize)^2),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t=1, trsize do 
   trainDataB.data[t] = image.scale(trainData.data[t]:float(), olinsize, olinsize)
   xlua.progress(t, trsize)
end
for t=1, tesize do 
   testDataB.data[t] = image.scale(testData.data[t]:float(), olinsize, olinsize)
   xlua.progress(t, tesize)
end

-- relocate pointers to new dataset:
trainDataIN = trainData -- save original dataset
testDataIN = testData
trainData = trainDataB -- relocate new dataset
testData = testDataB

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize,3,olinsize,olinsize)
testData.data = testData.data:reshape(tesize,3,olinsize,olinsize)


----------------------------------------------------------------------
print "==> creating network:"

olnet = torch.load('encoder_new.net')
--return  olnet:forward(trainData.data[1]:float()) -- returns a 128x1x1 output


----------------------------------------------------------------------
print "==> processing dataset with new network"

-- processing dataset:
trainData2 = {
   data = torch.Tensor(trsize, 128),
   labels = trainData.labels:clone(),
   size = function() return trsize end
}
testData2 = {
   data = torch.Tensor(tesize, 128),
   labels = testData.labels:clone(),
   size = function() return tesize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = olnet:forward(trainData.data[t]:float())
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size() do
   testData2.data[t] = olnet:forward(testData.data[t]:float())
   xlua.progress(t, testData:size())
end

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2


----------------------------------------------------------------------
print "==> creating 2-layer network classifier"

outsize = torch.max(trainData.labels)
nhiddens = 1024

model = nn.Sequential()
--model:add(nn.Reshape(nk*nk*(olnetoutsize)^2))
model:add(nn.Linear(128,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,outsize))

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










