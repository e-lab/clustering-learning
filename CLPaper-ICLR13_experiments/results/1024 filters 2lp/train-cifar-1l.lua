----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 1st layer test
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
cmd:option('-images', 'images', 'directory full of images')
cmd:option('-maximages', 100, 'max nb of images')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches') -- 9x9 kernels wanted
cmd:option('-nkernels', 1024, 'number of kernels to learn')
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

is = params.inputsize
nk = params.nkernels

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

trsize = 50000
tesize = 2000

trainData = {
   data = torch.Tensor(trsize, 3*32*32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('../datasets/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('../datasets/cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
--trsize = 5000  -- repeated here for smaller size train/test
--tesize = 200
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()


----------------------------------------------------------------------
print "==> preparing images"

print '==> whitening images (local normalization)'
highpass = image.gaussian1D(9)
n1 = nn.SpatialContrastiveNormalization(1,highpass)
for i=1,trainData.data:size(1) do 
   for c = 1,3 do
      trainData.data[i][c] = n1(trainData.data[i][{{c}}])
   end
end

for i=1,testData.data:size(1) do 
   for c = 1,3 do
      testData.data[i][c] = n1(testData.data[i][{{c}}])
   end
end

print '==> extracting patches'
data = torch.Tensor(params.nsamples,3*is*is)
for i = 1,params.nsamples do
   local img = math.random(1,trainData.data:size(1))
   local image = trainData.data[img]
   local x = math.random(1,trainData.data:size(3)-is+1)
   local y = math.random(1,trainData.data:size(4)-is+1)
   local randompatch = image[{ {},{y,y+is-1},{x,x+is-1} }]
   data[i] = randompatch
end

if not paths.filep('cifar10-1l.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk,3,is,is),
                          padding=2, symmetric=true, zoom=2, win=win,
                          nrow=math.floor(math.sqrt(nk)),
                          legend='K-Means Centroids'}
      end
   end                    
   kernels = unsup.kmeans(data, nk, params.initstd,params.niter, params.batchsize,cb,true)
   print('==> saving centroids to disk:')
   torch.save('cifar10-1l.t7', kernels)
else
   print '==> loading pre-trained k-means kernels'
   kernels = torch.load('cifar10-1l.t7')
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
cvstepsize = 2
poolsize = 2
outsize = torch.max(trainData.labels)

l1net = nn.Sequential()
l1net:add(nn.SpatialConvolution(3, nk, is, is, cvstepsize, cvstepsize))
l1net:add(nn.Tanh())
l1net:add(nn.SpatialLPPooling(nk, 2, poolsize, poolsize, poolsize, poolsize))
l1net:add(nn.SpatialSubtractiveNormalization(nk, normkernel))
-- output is a nk * 2 * 2 
l1netoutsize = o1size/poolsize/cvstepsize

-- initialize 1st layer parameters to learned filters:
for i=1,nk do   
   l1net.modules[1].weight[i] = kernels[i]:reshape(3, is, is):type('torch.FloatTensor')
end
l1net.modules[1].bias = l1net.modules[1].bias *0

--tests:
--td_1=torch.zeros(3,32,32)
--print(l1net:forward(td_1)[1])

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
--print "==> creating 1-layer network classifier"
--
--model = nn.Sequential()
--model:add(nn.Reshape(nk*(o1size/poolsize)^2))
--model:add(nn.Linear(nk*(o1size/poolsize)^2, outsize))

print "==> creating 2-layer network classifier"

nhiddens = outsize^2

model = nn.Sequential()
model:add(nn.Reshape(nk*(l1netoutsize)^2))
model:add(nn.Linear(nk*(l1netoutsize)^2,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,outsize))


print "==> test network output:"
print(model:forward(trainData.data[1]))

--loss = nn.MultiMarginCriterion()
dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end



