----------------------------------------------------------------------
-- Author: Eugenio Culurciello, Aysegul Dundar
-- This script is 2layer unsupervised network with clustering learning
-- This code is used for the paper : http://arxiv.org/abs/1306.0152

----------------------------------------------------------------------

require 'pl'
require 'image'
require 'nnx'
require 'optim'
require 'trainLayer' -- functions for Clustering Learning on video
require 'whiten'
require 'unsup'      -- standard kmeans

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>> Clustering learning on multiple datasets <<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 0)           L2 penalty on the weights
   -m,--momentum           (default 0)           momentum
   -b,--batchsize          (default 1000)        batch size for k-mean
   -t,--threads            (default 8)           number of threads
   -s,--size               (default extra)       dataset: small or full or extra
   -o,--save               (default results)     save directory
   -c,--colorbypass        (default true)        subsampling of the input and feeding into the classifier
   -a,--nsamples           (default 40000)       samples for the kmeans
   -l,--plot               (default true)        plot training/testing curves
   -e,--niter              (default 15)          number of iterations for k-means
   -n,--loss               (default nll)         type of loss function to minimize: nll | mse | margin
   -w,--whitening          (default true)        whitening applied to first layer
   -f,--dataset            (default cifar)       dataset: cifar or svhn
]]
opt.initstd= 0.1
opt.batchSize = 1 -- mini batch for the stochastic gradient
verbose = false    -- display information and kernels
if (opt.whitening=='false') then opt.whitening = false end  -- false from the option is not boolean format
torch.setdefaulttensortype('torch.FloatTensor')
opt.threads = tonumber(opt.threads)
if opt.threads > 1 then
   torch.setnumthreads(opt.threads)
   print('<trainer> using ' .. opt.threads .. ' threads')
end

----------------------------------------------------------------------
-- loading and processing dataset:
if opt.dataset=='cifar' then
   dofile '1_data_cifar.lua'
else 
   dofile '1_data_svhn.lua'
end

-- input image dateaset params:
ivch = trainData.data[1]:size(1) -- channels

trainData.data = trainData.data
testData.data = testData.data

----------------------------------------------------------------------
print '==> generating CL unsupervised network:'

-- compute network CL train time
time = sys.clock()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
kernels1, counts1, M, P = trainLayer(1, trainData.data, opt.nsamples, nk1, is1, verbose)
   
---------------------------------------------------------------------- 
--  1st layer network
opt.model = '1st-layer'
dofile '2_model.lua'

-- setup net/ load kernels into network:
model.modules[1].bias = model.modules[1].bias*0 -- set bias to 0
model.modules[1].weight = kernels1:reshape(nk1, ivch, is1,is1)

----------------------------------------------------------------------
print '==> process dataset throught 1st layer:'
if opt.whitening then 
   trainData2, testData2 = whitenprocessLayer(model, trainData.data, testData.data, M, P, verbose)
else
   trainData2, testData2 = processLayer(model, trainData.data, testData.data, verbose)
end

----------------------------------------------------------------------
print '==> Computing connection tables based on co-occurence of features and generate filters'
cTable2, kernels2 = createCoCnx(2, trainData2, nk1, feat_group, fanin, opt.nsamples, is2, verbose)
nk2 = cTable2:max()

----------------------------------------------------------------------
-- 2nd layer
opt.model = '2nd-layer'
dofile '2_model.lua'

-- setup net/ load kernels into network:
model.modules[1].bias = model.modules[1].bias*0 -- set bias to 0
model.modules[1].weight = kernels2:reshape(kernels2:size(1),is2,is2)  

----------------------------------------------------------------------
print '==> process dataset throught 2nd layer:'

trainData2, testData2 = processLayer(model, trainData2, testData2, verbose)

----------------------------------------------------------------------
-- compute network creation time time 
time = sys.clock() - time
print("<net> time to CL train network = " .. (time*1000) .. 'ms')

-- colorbypass
model = nn.Sequential()
model:add(nn.SpatialDownSampling(ss3,ss3,ss3,ss3))
trainData3, testData3 = processLayer(model, trainData.data, testData.data, verbose)

l1netoutsize = testData2:size(2)*testData2:size(3)*testData2:size(4)
cdatasize = trainData3:size(2)*trainData3:size(3)*trainData3:size(4) 
-- concatenate final network output  
trainData.data = torch.cat(trainData3:reshape(trsize, cdatasize), 
				trainData2:reshape(trsize, l1netoutsize)):float()
testData.data = torch.cat(testData3:reshape(tesize, cdatasize), 
				testData2:reshape(tesize, l1netoutsize)):float()

----------------------------------------------------------------------
-- classifier for train/test:
----------------------------------------------------------------------
print "==> creating classifier"
opt.model = '2mlp-classifier'
dofile '2_model.lua' 


dofile '3_loss.lua'  
dofile '4_train.lua'
dofile '5_test.lua'
   
----------------------------------------------------------------------
print "==> training classifier"
   
while true do
   train()
   test()
end
   
