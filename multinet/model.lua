----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'Dropout'

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

local CNN = nn.Sequential()
CNN:add(nn.SpatialConvolutionMM(ivch, nk1, is1, is1))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
-- 2nd layer 
CNN:add(nn.SpatialConvolutionMM(nk1,nk2, is2, is2))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))
-- 3rd layer
CNN:add(nn.SpatialConvolutionMM(nk2,nk3, is3, is3))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss3,ss3,ss3,ss3))

----------------------------------------------------------------------

-- compute network creation time time
time = sys.clock() - time
print("==>  time to CL train network = " .. (time*1000) .. 'ms')

----------------------------------------------------------------------
-- Classifier (trainable with mini-batch)
-- a 2-layer perceptron
local classifier = nn.Sequential()
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
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end
for _,layer in ipairs(classifier.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #classifier.modules-1 then
         layer.bias:zero()
      end
   end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

----------------------------------------------------------------------
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
   dropout = dropout
}

