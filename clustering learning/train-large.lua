----------------------------------------------------------------------
-- Run k-means on large dataset - learn 1st layer filters
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches') -- 9x9 kernels wanted
cmd:option('-nkernels', 64, 'number of kernels 1st layer')
cmd:option('-nkernels2', 256, 'number of kernels 2nd layer')
cmd:option('-niter', 25, 'nb of k-means iterations')
cmd:option('-niter2', 12, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 100*1000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-initstd2', 0.025, 'standard deviation to generate random initial templates')
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

is = opt.inputsize
nk1 = opt.nkernels
nk2 = opt.nkernels2

----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data.lua'

filename = paths.basename(opt.datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. opt.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, opt.inputsize)
dataset:conv()

trsize = dataset:size()

trainData = {
   data = torch.Tensor(trsize, 3, dataset[1][3]:size(1), dataset[1][3]:size(2)),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData.data[t] = dataset[i][3]:reshape(1,56,56):resize(3,56,56)
   xlua.progress(t, trainData:size())
end


----------------------------------------------------------------------
print '==> generating 1st layer filters:'
print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   local img = math.random(1,dataset:size())
   local image = dataset[i][3]
   local x = math.random(1,dataset[1][3]:size(1)-is+1)
   local y = math.random(1,dataset[1][3]:size(2)-is+1)
   local randompatch = image[{{y,y+is-1},{x,x+is-1} }]
   data[i] = randompatch
end

-- show a few patches:
f256S = data[{{1,256}}]:reshape(256,9,9)
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}

--if not paths.filep('berkeley56x56-1l.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk1,is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
      end
   end                    
   kernels = unsup.kmeans(data, nk1, opt.initstd,opt.niter, opt.batchsize,cb,true)
   print('==> saving centroids to disk:')
   torch.save('berkeley56x56-1l.t7', kernels)
--else
--   print '==> loading pre-trained k-means kernels'
--   kernels = torch.load('berkeley56x56-1l.t7')
--end

-- there is a bug in unpus.kmeans: some kernels come out nan!!!
-- clear nan kernels
for i=1,nk1 do   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end

-- normalize kernels to 0 mean and 1 std:
mean = {}
std = {}
for i=1,nk1 do
   mean[i] = kernels[i]:mean()
   std[i] = kernels[i]:std()
   kernels[i]:add(-mean[i])
   kernels[i]:div(std[i])
end

----------------------------------------------------------------------
print "==> loading and initialize 1 layer CL model"

opt.model = '1st-layer'
dofile '2_model.lua' 
l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
l1net.modules[1].weight = kernels:reshape(nk1,1,is,is):expand(nk1,3,is,is):type('torch.DoubleTensor')
l1net.modules[1].bias = l1net.modules[1].bias *0

--tests:
--td_1=torch.zeros(3,32,32)
--print(l1net:forward(td_1)[1])


----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData2 = {
   data = torch.Tensor(trsize, nk1*(l1netoutsize)^2),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trainData:size() do
   trainData2.data[t] = l1net:forward(trainData.data[t]:double())
   xlua.progress(t, trainData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())

trainData2.data = trainData2.data:reshape(trsize, nk1, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
trainData = trainData2 -- relocate new dataset

-- show a few outputs:
first256Samples = trainData2.data[{ {1,256},1 }]
image.display{image=first256Samples, nrow=16, nrow=16, padding=2, zoom=2, legend='Output 1st layer: first 256 examples'}


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------



----------------------------------------------------------------------
print '==> generating 2nd layer filters:'
print '==> extracting patches'
data2 = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   local img = math.random(1,trainData.data:size(1))
   local image = trainData.data[img]
   local z = math.random(1,trainData.data:size(2))
   local x = math.random(1,trainData.data:size(3)-is+1)
   local y = math.random(1,trainData.data:size(4)-is+1)
   local randompatch = image[{ {z},{y,y+is-1},{x,x+is-1} }]
   data2[i] = randompatch
end

-- show a few patches:
f256S2 = data2[{{1,256}}]:reshape(256,9,9)
image.display{image=f256S2, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}

--if not paths.filep('berkeley56x56-2l.t7') then
   print '==> running k-means'
   function cb (kernels2)
      if opt.visualize then
         win2 = image.display{image=kernels2:reshape(nk2,is,is), padding=2, symmetric=true, zoom=2, win=win2,
                          nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'}
      end
   end                    
   kernels2 = unsup.kmeans(data2, nk2, opt.initstd2, opt.niter2, opt.batchsize,cb,true)
   print('==> saving centroids to disk:')
   torch.save('berkeley56x56-2l.t7', kernels2)
--else
--   print '==> loading pre-trained k-means kernels'
--   kernels2 = torch.load('berkeley56x56-2l.t7')
--end

-- there is a bug in unpus.kmeans: some kernels come out nan!!!
-- clear nan kernels
for i=1,nk2 do   
   if torch.sum(kernels2[i]-kernels2[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels2[i] = torch.zeros(kernels2[1]:size()) 
   end
end

-- normalize kernels to 0 mean and 1 std:
mean = {}
std = {}
for i=1,nk2 do
   mean[i] = kernels2[i]:mean()
   std[i] = kernels2[i]:std()
   kernels2[i]:add(-mean[i])
   kernels2[i]:div(std[i])
end




