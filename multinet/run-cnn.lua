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
--require 'Dropout'

----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.2)                          learning rate
   -l,--learningRateDecay  (default 1e-7)                         learning rate decay (in # samples)
   -d,--dropout            (default 0.5)                          dropout amount
   -w,--weightDecay        (default 1e-5)                         L2 penalty on the weights
   -m,--momentum           (default 0.5)                          momentum
   -b,--batchSize          (default 128)                          batch size
   -t,--threads            (default 8)                            number of threads
   -p,--type               (default float)                        float or cuda
   -i,--devid              (default 1)                            device ID (if using CUDA)
   -s,--save               (default results/)                     file name to save network [after each epoch]
      --plot               (default true)                         plot error/accuracy live (if false, still logged in a file)
      --log                (default true)                         log the whole session to a file
      --seed               (default 1)                            use fixed seed for randomized initialization
      --German                                                    use the German road sign dataset
      --lim                (default 50)                           at least <lim> examples per sign, max 1000
      --maxBg              (default 1e9)                          max number of background samples
      --samplepercar       (default 6)                            number of the patch to extract from per car (bounding box)
      --visualize                                                 show some outputs while loading the datasets
      --dataStop                                                  use German stop signs
      --kittiRatio         (default 0.8)                          train/tot ratio for kitti dataset
      --siftflow                                                  load the siftflow dataset instead:noh
      --preproc            (default 'norm(y)+norm(u)+norm(v)')    preprocessing type, can be one of : norm(rgb) OR norm(yuv) OR norm(y)+norm(u)+norm(v) OR norm(y)+uv OR norm(y) OR rgb OR yuv
      --includeSiftflow                                           include siftflow dataset among other datasets
      --small                                                     uses a reduced version of the dataset (no shifting sample multiplication)
      --smallSift          (default true)                         reduced categories of the siftflow
]]

--[[dname,fname = sys.fpath()
parsed = tostring({'--nfeatures','--kernelsize','--subsize','--pooling','--hiddens','--slacmodel','--cnnmodel'})
opt.save = opt.save:gsub('PARAMS', parsed)]]

print('Options:')
print('  + learningrate:     ', opt.learningrate     )
print('  + learningRateDecay:', opt.learningRateDecay)
print('  + dropout:          ', opt.dropout          )
print('  + weightDecay:      ', opt.weightDecay      )
print('  + momentum:         ', opt.momentum         )
print('  + batchSize:        ', opt.batchSize        )
print('  + threads:          ', opt.threads          )
print('  + type:             ', opt.type             )
print('  + devid:            ', opt.devid            )
print('  + save:             ', opt.save             )
print('  + plot:             ', opt.plot             )
print('  + log:              ', opt.log              )
print('  + seed:             ', opt.seed             )
print('  + German:           ', opt.German           )
print('  + lim:              ', opt.lim              )
print('  + maxBg:            ', opt.maxBg            )
print('  + samplepercar:     ', opt.samplepercar     )
print('  + visualize:        ', opt.visualize        )
print('  + dataStop:         ', opt.dataStop         )
print('  + kittiRatio:       ', opt.kittiRatio       )
print('  + siftflow:         ', opt.siftflow         )
print('  + preproc:          ', opt.preproc          )
print('  + includeSiftflow   ', opt.includeSiftflow  )
print('  + small             ', opt.small            )
print('  + smallSift         ', opt.smallSift        )

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
print('Training: using ' .. opt.threads .. ' threads')

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
   print(cutorch.getDeviceProperties(opt.devid))
end

if opt.log then
   xlua.log(sys.dirname(opt.save) .. '/session.txt')
end

----------------------------------------------------------------------
-- load/get dataset
print '==> Loading datasets'
if opt.siftflow then
   require 'load-siftflow'
else
   require 'load-datasets'
end

print '==> Loading labels'
require 'load-labels'

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
