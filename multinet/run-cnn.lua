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
require 'Dropout'
--require 'online-kmeans' -- allow you to re-train k-means kernels
--[[require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require 'unsup']]

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.2)        learning rate
   -l,--learningRateDecay  (default 1e-7)       learning rate decay (in # samples)
   -d,--dropout            (default 0.5)        dropout amount
   -w,--weightDecay        (default 1e-5)       L2 penalty on the weights
   -m,--momentum           (default 0.5)        momentum
   -b,--batchSize          (default 128)        batch size
   -t,--threads            (default 8)          number of threads
   -p,--type               (default float)      float or cuda
   -i,--devid              (default 1)          device ID (if using CUDA)
   -s,--save               (default results/)   file name to save network [after each epoch]
      --plot               (default true)       plot error/accuracy live (if false, still logged in a file)
      --log                (default true)       log the whole session to a file
      --seed               (default 1)          use fixed seed for randomized initialization
      --German                                  use the German road sign dataset
      --lim                (default 50)         at least <lim> examples per sign, max 1000
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


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
print('Training: using ' .. opt.threads .. ' threads')

if opt.log then
   xlua.log(sys.dirname(opt.save) .. '/session.txt')
end

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
end

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
-- load/get dataset
print '==> Loading datasets'
require 'load-datasets'

local signLabels
if opt.German then
   signLabels = {
      '20 km/h speed limit',
      '30 km/h speed limit',
      '50 km/h speed limit',
      '60 km/h speed limit',
      '70 km/h speed limit',
      '80 km/h speed limit',
      '80 km/h end of speed limit',
      '100 km/h speed limit',
      '120 km/h speed limih',
      'No passing',
      'No passing for vehicles over 3.5t',
      'Priority',
      'Priority road',
      'Yield',
      'Stop',
      'Prohibited for all vehicles',
      'Vehicles over 3.5t prohibited',
      'Do not enter',
      'General danger',
      'Curve (left)',
      'Curve (right)',
      'Double curve. First curve is to the left',
      'Rough road',
      'Slippery when wet or dirty',
      'Road narrows (right side)',
      'Road work',
      'Traffic signals ahead',
      'Pedestrians',
      'Watch for children',
      'Bicycle crossing',
      'Beware of ice/snow',
      'Wild animal crossing',
      'End of all restrictions',
      'All traffic must turn right',
      'All traffic must turn left',
      'All traffic must continue straight ahead (i.e. no turns)',
      'All traffic must continue straight ahead or turn right (i.e. no left turn)',
      'All traffic must continue straight ahead or turn left (i.e. no right turn)',
      'Pass by on right',
      'Pass by on left',
      'Roundabout',
      'End of no passing zone',
      'End of no passing zone for vehicles over 3.5t'
   }
else
   signLabels = {
      'Stop',
      'Signal ahead',
      'Pedestrian crossing',
      'Speed limit 35',
      'Keep right',
      'Speed limit urdbl',
      'Merge',
      'School',
      'Speed limit 25',
      'Added lane',
      'Stop ahead',
      'Speed limit 40',
      'Speed limit 45',
      'Yield ahead',
      'School speed limit 25',
      'Speed limit 50',
      'Yield',
      'Right lane must turn',
      'Turn right',
      'Truck speed limit 55',
      'Lane ends',
      'Speed limit 65',
      'Ramp speed advisory 50',
      'Ramp speed advisory 45',
      'Do not pass',
      'No left turn',
      'Slow',
      'Zone ahead 45',
      'No right turn',
      'Turn left',
      'Ramp speed advisory urdbl',
      'Dip',
      'Speed limit 55',
      'Intersection',
      'Ramp speed advisory 20'
   }
end

classes = {'Person'}
for i = 1, nbClasses[2]  do classes[#classes+1] = signLabels[i] end
classes[#classes+1] = 'Background'
classes[#classes+1] = 'Car'

----------------------------------------------------------------------
-- define network to train

print('<trainer> creating new network')

nnf1,nnf2,nnf3  = 1,1,1             -- number of frames at each layer
nk0,nk1,nk2,nk3 = 3,16,32,64      -- nb of features
is0,is1,is2,is3 = 15,7,5,5          -- size of kernels
ss1,ss2,ss3     = 2,2,4               -- size of subsamplers (strides)
scales          = 1                 -- scales
fanin           = 8                 -- createCoCnxTable creates also 2*fanin connections
feat_group      = 32                --features per group (32=best in CIFAR, nk1=32, fanin=2)
opt.hiddens     = 64               -- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2   = nk3, opt.hiddens  -- dimensions for top perceptron
ivch            = 3

-- dropout?
local dropout = nn.Dropout(opt.dropout)

----------------------------------------------------------------------

print '==> generating CNN network:'

-- compute network CL train time
time = sys.clock()

CNN = nn.Sequential()
CNN:add(nn.SpatialConvolutionMM(ivch, nk1, is1, is1)) -- TODO SpatialConvolutionMM!!
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
-- 2nd layer -- TODO fanin 8, random connections
CNN:add(nn.SpatialConvolutionMM(nk1,nk2, is2, is2)) -- connex table based on similarity of features
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))
-- 3rd layer
CNN:add(nn.SpatialConvolutionMM(nk2,nk3, is3, is3)) -- connex table based on similarity of features
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss3,ss3,ss3,ss3))

----------------------------------------------------------------------

-- compute network creation time time
time = sys.clock() - time
print("==>  time to CL train network = " .. (time*1000) .. 'ms')

----------------------------------------------------------------------
-- Classifier (trainable with mini-batch)
-- a 2-layer perceptron
classifier = nn.Sequential()
classifier:add(nn.Reshape(cl_nk1))
classifier:add(nn.Linear(cl_nk1,cl_nk2))
classifier:add(nn.Threshold())
classifier:add(dropout)
classifier:add(nn.Linear(cl_nk2,#classes))

-- final stage: log probabilities
classifier:add(nn.LogSoftMax())

-- putting network together: a <Sequential> of <Sequential>s
--    <model>
--       |___<CNN>
--       |___<classifier>

-- adjust all biases for threshold activation units
for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:add(.1)
   end
end
for _,layer in ipairs(classifier.modules) do
   if layer.bias then
      layer.bias:add(.1)
   end
end

model = nn.Sequential()
model:add(CNN)
model:add(classifier)

--[[ Save model (pointless here, I will save it after training)
if opt.save then
   print('==>  <trainer> saving bare network to '..opt.save)
   os.execute('mkdir -p ' .. opt.save)
   torch.save(opt.save..'network.net', model)
end]]

-- verbose
print('==>  model:')
print(model)

----------------------------------------------------------------------
-- Loss: NLL
loss = nn.ClassNLLCriterion()

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

----------------------------------------------------------------------

print '==> load modules'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print '==> training!'

while true do
   train(trainData)
   test(testData)
end
