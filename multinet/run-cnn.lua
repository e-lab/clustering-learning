----------------------------------------------------------------------
-- Test of clustering learning on INRIA pedestrian, etc datasets
-- April 2013
--
-- Author: Eugenio Culurciello, Feb 2013 for Clustering Learning
--
----------------------------------------------------------------------
-- TODO:
----------------------------------------------------------------------

-- CNN test for CNN only

require 'pl'
require 'image'
require 'nnx'
require 'optim'
require 'online-kmeans' -- allow you to re-train k-means kernels
require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require 'unsup'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.2)        learning rate
   -d,--learningRateDecay  (default 1e-7)       learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)       L2 penalty on the weights
   -m,--momentum           (default 0.5)        momentum
   -d,--dropout            (default 0.5)        dropout amount
   -b,--batchSize          (default 128)        batch size
   -t,--threads            (default 8)          number of threads
   -p,--type               (default float)      float or cuda
   -i,--devid              (default 1)          device ID (if using CUDA)
   -s,--size               (default extra)      dataset: small or full or extra
   -o,--save               (default results)    save directory
   -n,--network            (default false)      path to existing [trained] network
   -s,--save               (default scratch/)   file name to save network [after each epoch]
      --display            (default false)      display training/testing samples while training
      --plot               (default false)      plot error/accuracy live (if false, still logged in a file)
      --log                (default true)       log the whole session to a file
      --seed               (default 1)          use fixed seed for randomized initialization
]]

opt.quicktest = true     --(default 0)          true = small test, false = full code running
opt.cnnmodel = true      --(default 1)          true = convnet model with tanh and normalization, otherwise without
opt.videodata = true     --(default 1)          true = load video file, otherwise ??? data

opt.initstd = 0.1
opt.niter = 15
opt.kmbatchsize = 1000 -- kmeans batchsize


dname,fname = sys.fpath()
parsed = tostring({'--nfeatures','--kernelsize','--subsize','--pooling','--hiddens','--slacmodel','--cnnmodel'})
opt.save = opt.save:gsub('PARAMS', parsed)

if opt.type == 'float' then
   torch.setdefaulttensortype('torch.FloatTensor')
else
   torch.setdefaulttensortype('torch.DoubleTensor')
end

if opt.seed then
   torch.manualSeed(opt.seed)
end

if opt.log then
   xlua.log(sys.dirname(opt.save) .. '/session.txt')
end

opt.threads = tonumber(opt.threads)
if opt.threads > 1 then
   torch.setnumthreads(opt.threads)
   print('<trainer> using ' .. opt.threads .. ' threads')
end

----------------------------------------------------------------------
-- define network to train

print('<trainer> creating new network')

nnf1,nnf2,nnf3  = 1,1,1             -- number of frames at each layer
nk0,nk1,nk2,nk3 = 3,32,64,128       -- nb of features
is0,is1,is2,is3 = 15,7,7,7          -- size of kernels
ss1,ss2         = 2,2               -- size of subsamplers (strides)
scales          = 1                 -- scales
fanin           = 2                 -- createCoCnxTable creates also 2*fanin connections
feat_group      = 32                --features per group (32=best in CIFAR, nk1=32, fanin=2)
opt.hiddens     = 64                -- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2   = nk3, opt.hiddens  -- dimensions for top perceptron
classes         = {'person', 'bg'}  -- classes of objects to find

ivch=3

----------------------------------------------------------------------

print '==> generating CNN network:'

-- compute network CL train time
time = sys.clock()

model = nn.Sequential()
model:add(nn.SpatialConvolution(ivch, nk1, is1, is1))
model:add(nn.Threshold())
model:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
-- 2nd layer
model:add(nn.SpatialConvolution(nk1,nk2, is2, is2)) -- connex table based on similarity of features
model:add(nn.Threshold())
model:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))
-- 3rd layer
model:add(nn.SpatialConvolution(nk2,nk3, is3, is3)) -- connex table based on similarity of features

----------------------------------------------------------------------

-- compute network creation time time
time = sys.clock() - time
print("==>  time to CL train network = " .. (time*1000) .. 'ms')

----------------------------------------------------------------------
-- Classifier
-- a 2-layer perceptron
model:add(nn.Tanh())
model:add(nn.Reshape(cl_nk1))
model:add(nn.Linear(cl_nk1,cl_nk2))
model:add(nn.Tanh())
model:add(nn.Linear(cl_nk2,#classes))

-- final stage: log probabilities
model:add(nn.LogSoftMax())

-- Save model
if opt.save then
   print('==>  <trainer> saving bare network to '..opt.save)
   os.execute('mkdir -p "' .. sys.dirname(opt.save) .. '"')
   torch.save(opt.save..'network.net', model)
end

-- verbose
print('==>  model:')
print(model)

----------------------------------------------------------------------
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- load/get dataset
print "==> loading dataset:"
data  = require 'data-person'

print '==> load modules'
train = require 'train'
test  = require 'test'

----------------------------------------------------------------------
print '==> training!'

while true do
   train(data.trainData)
   test(data.testData)
end
