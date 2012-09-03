----------------------------------------------------------------------
-- Run k-means on SVHN dataset - learn 1st layer filters
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
cmd:option('-inputsize', 9, 'size of each input patches') -- kernels size wanted
cmd:option('-nkernels', 64, 'number of kernels to learn')
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

tar = 'http://data.neuflow.org/data/svhn.t7.tgz'

if not paths.dirp('../datasets/housenumbers') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = '../datasets/housenumbers/train_32x32.t7'
test_file = '../datasets/housenumbers/test_32x32.t7'
extra_file = '../datasets/housenumbers/extra_32x32.t7'


loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

loaded = torch.load(extra_file,'ascii')
extraTrainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

trsize = trainData.data:size(1)
tesize = testData.data:size(1)


----------------------------------------------------------------------
print '==> visualizing data'

--image.display{image=trainData.data[{ {1,256} }], nrow=16, legend='Some training samples'}
--image.display{image=extraTrainData.data[{ {1,256} }], nrow=16, legend='Extra training samples'}
--image.display{image=testData.data[{ {1,256} }], nrow=16, legend='Some test samples'}

----------------------------------------------------------------------
--print "==> resizing dataset to 46x46, required by online learner:"
--
--olinsize = 46 -- online learner network image input size
---- resize all dataset images for the online learner net:
--trainDataB = {
--   data = torch.Tensor(trsize, 3*(olinsize)^2),
--   labels = trainData.labels:clone(),
--   size = function() return trsize end
--}
--testDataB = {
--   data = torch.Tensor(tesize, 3*(olinsize)^2),
--   labels = testData.labels:clone(),
--   size = function() return tesize end
--}
--for t=1, trsize do 
--   trainDataB.data[t] = image.scale(trainData.data[t]:float(), olinsize, olinsize)
--   xlua.progress(t, trsize)
--end
--for t=1, tesize do 
--   testDataB.data[t] = image.scale(testData.data[t]:float(), olinsize, olinsize)
--   xlua.progress(t, tesize)
--end
--
---- relocate pointers to new dataset:
--trainDataIN = trainData -- save original dataset
--testDataIN = testData
--trainData = trainDataB -- relocate new dataset
--testData = testDataB
--


----------------------------------------------------------------------
print "==> preparing images"

-- reshape data                                                                                     
trainData.data = trainData.data:float()
testData.data = testData.data:float()

image.display{image=trainData.data[{ {1,256} }], nrow=16, legend='Some normalized training samples'}
image.display{image=testData.data[{ {1,256} }], nrow=16, legend='Some normalized test samples'}


print '==> whitening images (local normalization)'
highpass = image.gaussian1D(9)
n1 = nn.SpatialContrastiveNormalization(1,highpass)
for i=1,trsize do 
   for c = 1,3 do
      trainData.data[i][c] = n1(trainData.data[i][{{c}}]:double())
   end
   xlua.progress(i, trsize)
end

for i=1,tesize do 
   for c = 1,3 do
      testData.data[i][c] = n1(testData.data[i][{{c}}]:double())
   end
   xlua.progress(i, tesize)
end

image.display{image=trainData.data[{ {1,256} }], nrow=16, legend='Some normalized training samples'}
image.display{image=testData.data[{ {1,256} }], nrow=16, legend='Some normalized test samples'}

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


----------------------------------------------------------------------
print '==> running k-means'
function cb (kernels)
   if params.visualize then
      win = image.display{image=kernels:reshape(nk,3,is,is),
                       padding=2, symmetric=true, zoom=2, win=win,
                       nrow=math.floor(math.sqrt(nk)),
                       legend='K-Means Centroids'}
   end
end                    
kernels = unsup.kmeans(data, nk, params.initstd, params.niter, params.batchsize, cb, true)
print('==> saving centroids to disk:')
torch.save('SVHN-1l.t7', kernels)


-- there is a bug in unpus.kmeans: some kernels come out nan!!!
-- clear nan kernels
for i=1,nk do   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end

--print '==> loading filters'
--kernels = torch.load('SVHN-1l-64.t7')


------------------------------------------------------------
print "==> processing dataset with k-means kernels + pooling"

opt.model = '1st-layer'
dofile '2_model.lua' 

-- initialize 1st layer parameters to learned filters:
for i=1,nk do   
   l1net.modules[1].weight[i] = kernels[i]:reshape(3, is, is):double()
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
for t = 1,trainData:size(1) do
   trainData2.data[t] = l1net:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
for t = 1,testData:size(1) do
   testData2.data[t] = l1net:forward(testData.data[t]:double())
   xlua.progress(t, testData:size())
end

trainData2.data = trainData2.data:reshape(trsize, nk, l1netoutsize, l1netoutsize)
testData2.data = testData2.data:reshape(tesize, nk, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2

-- show a few outputs:
image.display{image=trainData.data[{ {1,256},{1} }], nrow=16, padding=2, zoom=2, legend='Processed dataset output'}

--------------------------------------------------------------------
print "==> creating 2-layer network classifier"
opt.model = '2mlp-classifier'
dofile '2_model.lua'

print "==> test network output:"
print(model:forward(trainData.data[1]))

dofile '3_loss.lua' 
dofile '4_train.lua'
dofile '5_test.lua'

------------------------------------------------------------------
print "==> training 1-layer network classifier"

while true do
   train()
   test()
end

-- save datasets:
--testData.data=testData.data:float()
--trainData.data=trainData.data:float() 
--torch.save('trainData-SVHN-l1-xxx.t7', trainData)
--torch.save('testData-SVHN-l1-xxx.t7', testData)


