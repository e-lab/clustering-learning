----------------------------------------------------------------------
-- Massive online trained network on videos
-- load all sort of video, run Clustering learning, online-learn forever
-- January 18th 2013, E. Culurciello with discussion w/ Clement Farabet
--
-- 1. load a video
-- 2. for each few frames: extract patches, cluster-learn filter
-- 3. setup net layer layer, process video through layer, then repeat step 2,3 for all layers
-- 4. periodically check what is learned: test on dataset?
-- 5. enjoy the ultimate net - Yay!
--
----------------------------------------------------------------------

-- TODO: extend k-means to multiple "winners" = average on multiple kernels
-- TODO: create NMaxPool layer: propagate multiple max as winners or average a few of them
-- TODO: group features for pooling
-- TODO: volumetric nn.Tanh, nn.pooling, etc, so we can add more volumeteric layers


require 'nnx'
require 'eex'
require 'image'
--require 'kmec'
--require 'unsup'
require 'online-kmeans'
require 'ffmpeg'
require 'trainLayer'
require 'optim'
require "slac"


cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches')
cmd:option('-nkernels', 32, 'number of kernels to learn')
cmd:option('-niter', 15, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 100*1000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:option('-savedataset', false, 'save modified dataset')
cmd:option('-classify', true, 'run classification train/test')
cmd:option('-nnframes', 1, 'nb of frames uses for temporal learning of features')
cmd:option('-dataset', '../datasets/faces_cut_yuv_32x32/','path to FACE dataset root dir')
cmd:option('-patches', 'all', 'nb of patches to use')
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
cmd:option('-type', 'double', 'type: double | float | cuda')

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
nnf1 = opt.nnframes

print 'SUPER-NET script!'
----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

--dspath = '/Users/eugenioculurciello/Pictures/2013/1-13-13/VID_20130105_111419.mp4'
--source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=false, load=false}

--dspath = '/Users/eugenioculurciello/Desktop/driving1.mov'
--source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=false, load=false}

--dspath = '../datasets/TLD/06_car'
--source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=true, load=false}

--dspath = '../datasets/TLD/08_volkswagen'
--source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=true, load=false}

--dspath = '../datasets/TLD/09_carchase'
--source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=true, load=false}

dspath = '/Users/eugenioculurciello/Desktop/euge.mov'
source = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=false, load=false}

rawFrame = source:forward()
-- input video params:
ivch = rawFrame:size(1) -- channels
ivhe = rawFrame:size(2) -- height
ivwi = rawFrame:size(3) -- width
source.current = 1 -- rewind video frames

-- number of frames to process:
nfpr = 100 + nnf1 -- batch process size [video frames]

-- normalize and prepare dataset:
neighborhood = image.gaussian1D(9)
normalization = nn.SpatialContrastiveNormalization(ivch, neighborhood, 1e-3)

function createDataBatch()
   trainData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
   for i = 1, nfpr do -- just get a few frames to begin with
      procFrame = normalization:forward(rawFrame) -- full LCN!
      trainData[i] = procFrame
      rawFrame = source:forward()
   end
   return trainData
end

createDataBatch()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 1
--kernels1 = trainLayer(nlayer, trainData, nil, nk1, nnf1, is) --no slac
-- SLAC: nk1*4 filters to learn, then narrow down to nk1:
kernels1 = trainLayer(nlayer, trainData, nil, nk1*4, nnf1, is) -- with slac
-- kernels1, connTable1 = slac(kernels1, startN, finalN,tau,Delta) -- SLAC algorithm to aggregate kernels
kernels1s, connTable1 = slac(kernels1, nk1*4, nk1, is, is) -- SLAC algorithm to aggregate kernels
image.display{image=kernels1s:reshape(kernels1s:size(1),is,is), padding=2, symmetric=true, zoom=2} -- show organization
kernels1 = kernels1s--[{{1,nk1}}]
nk1s=kernels1s:size(1)

----------------------------------------------------------------------
print '==> create model 1st layer:'

poolsize = 2
cvstepsize = 1
normkernel = image.gaussian1D(7)
ovhe = (ivhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi = (ivwi-is+1)/poolsize/cvstepsize -- output video feature width

vnet = nn.Sequential()
-- usage: VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
vnet:add(nn.VolumetricConvolution(ivch, nk1s, nnf1, is, is, 1, cvstepsize,cvstepsize)) --SLAC with nk1s
vnet:add(nn.Sum(2))
vnet:add(nn.SpatialMaxMap(connTable1))
vnet:add(nn.Tanh())
vnet:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize))
vnet:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

-- load kernels into network:
kernels1:div(nnf1*nk1*ivch) -- divide kernels so output of SpatialConv is about ~1 or more
--vnet.modules[1].weight = kernels1:reshape(nk1,nnf1,is,is):reshape(nk1,1,nnf1,is,is):expand(nk1,ivch,nnf1,is,is)
vnet.modules[1].weight = kernels1:reshape(nk1s,nnf1,is,is):reshape(nk1s,1,nnf1,is,is):expand(nk1s,ivch,nnf1,is,is) -- SLAC


----------------------------------------------------------------------
print '==> process video throught 1st layer:'

function processLayer(lv, network, data_in, nkernels, oheight, owidth)
   data_out = torch.Tensor(nfpr, nkernels, oheight, owidth)
   for i = nnf1, nfpr do -- just get a few frames to begin with
      if lv == 1 then procFrames = data_in[{{i-nnf1+1,i},{},{}}]:transpose(1,2) -- swap order of indices here for VolConvolution to work
      else            procFrames = data_in[i] end
      data_out[i] = network:forward(procFrames)
      xlua.progress(i, nfpr)
      -- do a live display of the input video and output feature maps 
      winm = image.display{image=data_out[i], padding=2, zoom=1, win=winm, nrow=math.floor(math.sqrt(nkernels))}
   end
   -- data_out = nil --free memory if needed
   return data_out
end

trainData2 = processLayer(1, vnet, trainData, nk1, ovhe, ovwi)

--report some statistics:
print('1st layer max: '..vnet.modules[1].output:max()..' and min: '..vnet.modules[1].output:min()..' and mean: '..vnet.modules[1].output:mean())

----------------------------------------------------------------------
print '==> generating filters for layer 2:'
nlayer = 2
nnf2 = 1
nk2 = 64
kernels2 = trainLayer(nlayer, trainData2, nil, nk2, nnf2, is)


----------------------------------------------------------------------
print '==> create model 2nd layer:'

poolsize = 2
cvstepsize = 1
ovhe2 = (ovhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi2 = (ovwi-is+1)/poolsize/cvstepsize -- output video feature width
fanin = 8

vnet2 = nn.Sequential()
--vnet2:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin), is, is,cvstepsize,cvstepsize))
vnet2:add(nn.SpatialConvolution(nk1, nk2, is, is,cvstepsize,cvstepsize)) -- fully connected
vnet2:add(nn.Tanh())
vnet2:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize))
vnet2:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

-- load kernels into network:
kernels2:div(nk2) -- divide kernels so output of SpatialConv is about ~1 or more
vnet2.modules[1].weight = kernels2:reshape(nk2,is,is):reshape(nk2,1,is,is):expand(nk2,nk1,is,is)


----------------------------------------------------------------------
-- 2 layer test:
print '==> Test network'
dofile 'test-videoknet.lua'


----------------------------------------------------------------------
print '==> process video throught 2nd layer:'
print 'Initial frames will be blank because of the VolConv on 1st layer~'

trainData3 = processLayer(2, vnet2, trainData2, nk2, ovhe2, ovwi2)

--report some statistics:
print('2nd layer max: '..vnet2.modules[1].output:max()..' and min: '..vnet2.modules[1].output:min()..' and mean: '..vnet2.modules[1].output:mean())


----------------------------------------------------------------------
print '==> generating filters for layer 3:'
nlayer = 3
nnf3 = 1
nk3 = 128
kernels3 = trainLayer(nlayer, trainData2, nil, nk3, nnf3, is)

----------------------------------------------------------------------
print '==> create model 3nd layer:'

poolsize = 2
cvstepsize = 1
ovhe3 = (ovhe2-is+1)/poolsize/cvstepsize -- output video feature height
ovwi3 = (ovwi2-is+1)/poolsize/cvstepsize -- output video feature width

vnet3 = nn.Sequential()
vnet3:add(nn.SpatialConvolution(nk2, nk3, is, is,cvstepsize,cvstepsize))
vnet3:add(nn.Tanh())
vnet3:add(nn.SpatialLPPooling(nk3, 2, poolsize, poolsize, poolsize, poolsize))
vnet3:add(nn.SpatialSubtractiveNormalization(nk3, normkernel))

-- load kernels into network:
kernels3:div(nk3) -- divide kernels so output of SpatialConv is about ~1 or more
vnet3.modules[1].weight = kernels3:reshape(nk3,is,is):reshape(nk3,1,is,is):expand(nk3,nk2,is,is)


----------------------------------------------------------------------
-- 3 layer test:
print '==> Test network'
dofile 'test-videoknet.lua'


torch.load() -- break function


----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> Now test a few loops of online learning on video'


-- save older kernels to x-check online routines:
kernels1_old = kernels1:clone()
kernels2_old = kernels2:clone()

-- generate more samples:
source.current = source.current - nnf1 -- rewind video
createDataBatch()

-- update kernels with new data:
kernels1 = trainLayer(1, trainData, kernels1, nk1, nnf1, is)
kernels2 = trainLayer(2, trainData2, kernels2, nk2, nnf2, is)

processLayer1()

--report some statistics:
print('1st layer max: '..vnet.modules[1].output:max()..' and min: '..vnet.modules[1].output:min()..' and mean: '..vnet.modules[1].output:mean())

processLayer2()

--report some statistics:
print('2nd layer max: '..vnet2.modules[1].output:max()..' and min: '..vnet2.modules[1].output:min()..' and mean: '..vnet2.modules[1].output:mean())


-- show filters before and after new training:
--image.display{image=kernels1:reshape(nk1,nnf1*is,is), padding=2, symmetric=true,                                            zoom=2, nrow=math.floor(math.sqrt(nk1)), legend='Layer '..nlayer..' filters'}
--image.display{image=kernels1_old:reshape(nk1,nnf1*is,is), padding=2, symmetric=true,                                        zoom=2, nrow=math.floor(math.sqrt(nk1)), legend='Layer '..nlayer..' filters'}
--
--image.display{image=kernels2:reshape(nk2,nnf2*is,is), padding=2, symmetric=true,                                            zoom=2, nrow=math.floor(math.sqrt(nk2)), legend='Layer '..nlayer..' filters'}
--image.display{image=kernels2_old:reshape(nk2,nnf2*is,is), padding=2, symmetric=true,                                        zoom=2, nrow=math.floor(math.sqrt(nk2)), legend='Layer '..nlayer..' filters'}



