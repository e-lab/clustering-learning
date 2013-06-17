#!/usr/bin/env torch
----------------------------------------------------------------------
-- network for signs, pedestrians, cars - driving demo
-- June 2013
--
-- Author: Eugenio Culurciello, Alfredo Canziani
--
----------------------------------------------------------------------

require 'pl'
require 'image'
require 'nnx'
require 'optim'
require 'Dropout'

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
      --maxBg              (default 4170)       max number of background samples
      --samplepercar       (default 6)          number of the patch to extract from per car (bounding box)
]]

--[[dname,fname = sys.fpath()
parsed = tostring({'--nfeatures','--kernelsize','--subsize','--pooling','--hiddens','--slacmodel','--cnnmodel'})
opt.save = opt.save:gsub('PARAMS', parsed)]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
print('Training: using ' .. opt.threads .. ' threads')

if opt.log then
   xlua.log(sys.dirname(opt.save) .. '/session.txt')
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

print '==> load modules'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print '==> training!'

while true do
   train(trainData)
   test(testData)
end
