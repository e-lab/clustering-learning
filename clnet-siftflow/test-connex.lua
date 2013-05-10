-- Eugenio Culurciello, March 1st 2013
-- test conenction matrix for layers



--require 'xlua'
require 'image'
require 'nnx'
require 'optim'
-- ec added:
require 'eex'
require 'online-kmeans' -- allow you to re-train k-means kernels
require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require "slac"


----------------------------------------------------------------------
print '==> loading videt training-set'
dspath = '../datasets/driving1.mov'
--source = ffmpeg.Video{path=dspath, width = 120, height = 80, encoding='jpg', fps=24, loaddump=false, load=false}
source = ffmpeg.Video{path=dspath, width = 320, height = 240, encoding='jpg', fps=24, loaddump=false, load=false}

rawFrame = source:forward()
-- input video params:
ivch = rawFrame:size(1) -- channels
ivhe = rawFrame:size(2) -- height
ivwi = rawFrame:size(3) -- width
source.current = 1 -- rewind video frames

-- number of frames to process:
nfpr = 10 -- batch process size [video frames]


normthres = 1e-1
preproc = nn.Sequential()
preproc:add(nn.SpatialContrastiveNormalization(ivch, image.gaussian1D(7), normthres))

----------------------------------------------------------------------
-- normalize inout video data:

function createDataBatch()
   videoData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
   for i = 1, nfpr do -- just get a few frames to begin with
      procFrame = preproc:forward(rawFrame) -- full LCN!
      videoData[i] = procFrame
      rawFrame = source:forward()
   end
   return videoData
end

createDataBatch()



----------------------------------------------------------------------
----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> generating CL unsupervised network:'

-- compute network CL train time
   time = sys.clock()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 1
nnf1 = 1 -- number of frames from input video to use
nk1 = 16
nk = nk1
is = 7
opt={}
opt.nsamples = 1000  -- patch samples to use
opt.initstd = 0.1
opt.niter = 15
opt.batchsize = 1000
poolsize = 2
cvstepsize = 1
normkernel = image.gaussian1D(7)
ovhe = (ivhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi = (ivwi-is+1)/poolsize/cvstepsize -- output video feature width



kernels1p = trainLayer(nlayer, videoData, opt.nsamples, nil, nk1*2, nnf1, is, true) 
kernels1, connTable1 = slac(kernels1p, nk1*2, nk1, 5, 4.5) -- SLAC algorithm to aggregate kernels
-- image.display{image=kernels1:reshape(nk1/2,is,is), padding=2, zoom=2}

   
----------------------------------------------------------------------
-- 1st layer
   
   -- Trainable Network
   vnet = nn.Sequential()
   vnet:add(nn.SpatialConvolutionMap(nn.tables.random(ivch, nk1, 1), is, is, cvstepsize,cvstepsize))
   vnet:add(nn.Tanh())
   vnet:add(nn.SpatialMaxPooling(2,2,2,2))
   vnet:add(nn.SpatialContrastiveNormalization(nk1, normkernel,1e-3))

-- setup net/ load kernels into network:
vnet.modules[1].bias = vnet.modules[1].bias*0 -- set bias to 0!!! not needed
kernels1:div(nnf1*nk1*ivch/2) -- divide kernels so output of SpatialConv is about ~1 or more
vnet.modules[1].weight = kernels1:reshape(nk1,is,is) 

----------------------------------------------------------------------
print '==> process video throught 1st layer:'

function processLayer(lv, network, data_in, nkernels, oheight, owidth)
   data_out = torch.Tensor(nfpr, nkernels, oheight, owidth)
   for i = nnf1, nfpr do -- just get a few frames to begin with
      if ( nnf1>1 and lv == 1 ) then procFrames = data_in[{{i-nnf1+1,i},{},{}}]:transpose(1,2) -- swap order of indices here for VolConvolution to work
      else            procFrames = data_in[i] end
      data_out[i] = network:forward(procFrames)
      xlua.progress(i, nfpr)
      -- do a live display of the input video and output feature maps 
      if opt.display then
         winm = image.display{image=data_out[i], padding=2, zoom=1, win=winm, nrow=math.floor(math.sqrt(nkernels))}
      end
   end
   -- data_out = nil --free memory if needed
   return data_out
end

videoData2 = processLayer(nlayer, vnet, videoData, nk1, ovhe, ovwi)

--report some statistics:
print('1st layer conv out. Max: '..vnet.modules[1].output:max()..' and min: '..vnet.modules[1].output:min()..' and mean: '..vnet.modules[1].output:mean())
print('1st layer output. Max: '..vnet.output:max()..' and min: '..vnet.output:min()..' and mean: '..vnet.output:mean())



----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 2
nnf2 = 1 -- just one frames goes into layer 2
nk2 = 32
nk = nk2
is = 7
fanin = 2 -- number of connections per group
poolsize = 2
cvstepsize = 1
ovhe2 = (ovhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi2 = (ovwi-is+1)/poolsize/cvstepsize -- output video feature width



-- OUTPUT Co-occurence CONNEX MODEL:
print '==> Computing connection tables based on co-occurence of features'
cTable1 = createCoCnxTable(videoData2, nk1, nk2, fanin, 'AND') -- connex table based on co-occurence of features

-- train filter for next layer (kernels2) based on groups of cTable!!!
kernels2 = trainCoCnxLayer(nlayer, videoData2, cTable1, opt.nsamples*5, nk2, fanin, nnf2, is, false)


image.display{image=kernels2:reshape((#kernels2)[1],is,is), padding=2, symmetric=true, 
         zoom=2, nrow=8, legend='Layer 2 filters'}







