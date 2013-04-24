----------------------------------------------------------------------
-- Test of clustering learning on video of cars
-- April 2013
--
-- Author: Eugenio Culurciello, Feb 2013 for Clustering Learning
-- 
----------------------------------------------------------------------
-- TODO: 
----------------------------------------------------------------------

require 'pl'
require 'image'
require 'nnx'
require 'optim'
-- ec added:
--require 'eex'
require 'online-kmeans' -- allow you to re-train k-means kernels
require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require 'unsup'
--require "slac"

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
   -n,--network				(default false)		path to existing [trained] network
	-s,--save					(default scratch/) 	file name to save network [after each epoch]
	--display			(default false) 			display training/testing samples while training
	--plot 				(default false)			plot error/accuracy live (if false, still logged in a file)
	--log					(default true)				log the whole session to a file
	--preproc			(default norm(y)+uv)	preprocessing type, can be one of : norm(rgb) OR norm(yuv) OR norm(y)+norm(u)+norm(v) OR norm(y)+uv OR norm(y) OR rgb OR yuv
	--scales				(default "1")			scales
	--share				(default 0)					share weights across scales: "all" means all, 0 means none, ..1 means first layer, 2 first two layers...         
	--nfeatures			(default "3,32,64,128")	number of features, for each layer
	--kernelsize		(default "15,7,7,7")		kernel size, at each layer
	--subsize			(default "2,2")			subsampling size, at each layer
	--pooling			(default "max")			subsampling/pooling type: max OR sum OR l2
	--hiddens			(default "256")			nb of hidden features for top perceptron (0=linear classifier)
	--seed				(default 1)					use fixed seed for randomized initialization
]]

opt.quicktest = true	--(default 0)			true = small test, false = full code running
opt.cnnmodel = true --(default 1)			true = convnet model with tanh and normalization, otherwise without
opt.videodata = true --	(default 1) 		true = load video file, otherwise ??? data


dname,fname = sys.fpath()
parsed = tostring({'--nfeatures','--kernelsize','--subsize','--pooling','--hiddens',
							'--slacmodel','--cnnmodel'})
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
--
print('<trainer> creating new network')

pcall(loadstring("f0_,f1_,f2_,f3_ = " .. opt.nfeatures))         -- nb of features
pcall(loadstring("k0_,k1_,k2_,k3_ = " .. opt.kernelsize))           -- size of kernels
pcall(loadstring("s0_,s1_         = " .. opt.subsize))           -- size of subsamplers (strides)
pcall(loadstring("scales          = {" .. opt.scales .. "}"))    -- scales
trainable_ss    = s0_*s1_                                        -- global subsampling
trainable_fov   = ((k3_*s1_+k2_-1)*s0_+k1_-1)                    -- global field of view
patchSize       = trainable_fov                                  -- patch size is the same
p0_,p1_         = f3_*#scales,tonumber(opt.hiddens)              -- dimensions for top perceptron

normthres = 1e-1

-- Preprocessor (normalizer)
preproc = nn.Sequential()
if opt.preproc == 'norm(rgb)' then
	preproc:add(nn.SpatialContrastiveNormalization(f0_, image.gaussian1D(k0_), normthres))
elseif opt.preproc == 'norm(yuv)' then
	preproc:add(nn.SpatialColorTransform('rgb2yuv'))
	preproc:add(nn.SpatialContrastiveNormalization(f0_, image.gaussian1D(k0_), normthres))
elseif opt.preproc == 'norm(y)+norm(u)+norm(v)' then
	preproc:add(nn.SpatialColorTransform('rgb2yuv'))
	do
		normer = nn.ConcatTable()
		for i = 1,3 do
			local n = nn.Sequential()
			n:add(nn.Narrow(1,i,1))
			n:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
			normer:add(n)
		end
	end
	preproc:add(normer)
	preproc:add(nn.JoinTable(1))
elseif opt.preproc == 'norm(y)+uv' then
	preproc:add(nn.SpatialColorTransform('rgb2yuv'))
	do
		ynormer = nn.Sequential()
		ynormer:add(nn.Narrow(1,1,1))
		ynormer:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
		normer = nn.ConcatTable()
		normer:add(ynormer)
		normer:add(nn.Narrow(1,2,2))
	end
	preproc:add(normer)
	preproc:add(nn.JoinTable(1))
elseif opt.preproc == 'norm(y)' then
	f0_ = 1
	preproc:add(nn.SpatialColorTransform('rgb2y'))
	preproc:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
elseif opt.preproc == 'rgb' then
	preproc:add(nn.Identity())
elseif opt.preproc == 'yuv' then
	preproc:add(nn.SpatialColorTransform('rgb2yuv'))
else
	print('incorrect arg: preproc')
	op:help()
	os.exit()
end



----------------------------------------------------------------------
if opt.videodata then
   print '==> loading videt training-set'
   dspath = '../../datasets/driving1.mov'
   --source = ffmpeg.Video{path=dspath, width = 120, height = 80, encoding='jpg', fps=24, loaddump=false, load=false}
   source = ffmpeg.Video{path=dspath, width = 320, height = 240, encoding='jpg', fps=24, loaddump=false, load=false}
   
   rawFrame = source:forward()
   -- input video params:
   ivch = rawFrame:size(1) -- channels
   ivhe = rawFrame:size(2) -- height
   ivwi = rawFrame:size(3) -- width
   source.current = 1 -- rewind video frames

else 
   print '==> loading ??? training-set'
	--dofile('sf-dataset.lua')
   
   -- input image dateaset params:
   ivch = trainData[1][1]:size(1) -- channels
   ivhe = trainData[1][1]:size(2) -- height
   ivwi = trainData[1][1]:size(3) -- width
   
end


-- number of frames to process:
if opt.quicktest then nfpr = 10 -- batch process size [video frames]
else nfpr = 200 end



----------------------------------------------------------------------
-- normalize input video data:

function createDataBatch()
   videoData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
   for i = 1, nfpr do -- just get a few frames to begin with
      -- perform full LCN
      if opt.videodata then procFrame = preproc:forward(rawFrame) 
      else procFrame = preproc:forward(trainData[i][1]:clone())
      end
      videoData[i] = procFrame
      if opt.videodata then rawFrame = source:forward() end
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
nk1 = f1_
nk = nk1
is = k1_
if opt.quicktest then opt.nsamples = 300 else opt.nsamples = 10000 end  -- patch samples to use
opt.initstd = 0.1
opt.niter = 15
opt.batchsize = 1000
poolsize = 2
cvstepsize = 1
normkernel = image.gaussian1D(7)
ovhe = (ivhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi = (ivwi-is+1)/poolsize/cvstepsize -- output video feature width

 
-- AND/OR model or FULL CONNECT MODEL:
kernels1 = trainLayer(nlayer, videoData, opt.nsamples, nil, nk1, nnf1, is, false) 

if opt.display then image.display{image=kernels1:reshape(nk1,ivch,is,is),
		padding=2, symmetric=true, nrow = 8, zoom=2, legend = 'Layer 1 filters'} end

   
----------------------------------------------------------------------
-- 1st layer

vnet = nn.Sequential()
if nnf1 > 1 then
	vnet:add(nn.VolumetricConvolution(ivch, nk1, nnf1, is, is, 1, cvstepsize,cvstepsize))
	vnet:add(nn.Sum(2)) -- needed by volconv
elseif nnf1 == 1 then
	vnet:add(nn.SpatialConvolution(ivch, nk1, is, is, cvstepsize,cvstepsize))
end
vnet:add(nn.Threshold())
if opt.pooling == 'max' then
	vnet:add(nn.SpatialMaxPooling(s0_,s0_,s0_,s0_))
elseif opt.pooling == 'sum' then
	vnet:add(nn.SpatialSubSampling(f1_,s0_,s0_,s0_,s0_))
elseif opt.pooling == 'l2' then
	vnet:add(nn.SpatialLPPooling(f1_,2,s0_,s0_,s0_,s0_))
end

-- setup net/ load kernels into network:
vnet.modules[1].bias = vnet.modules[1].bias*0 -- set bias to 0!!! not needed
kernels1_ = kernels1:clone():div(nnf1*nk1) -- divide kernels so output of SpatialConv std ~0.5
vnet.modules[1].weight = kernels1_:reshape(nk1,ivch,is,is)


----------------------------------------------------------------------
print '==> process video throught 1st layer:'
videoData2 = processLayer(nlayer, vnet, videoData, nk1, ovhe, ovwi)

--report some statistics:
print('1st layer conv out. std: '..vnet.modules[1].output:std()..' and mean: '..vnet.modules[1].output:mean())
print('1st layer output. std: '..vnet.output:std()..' and mean: '..vnet.output:mean())



----------------------------------------------------------------------
print '==> generating filters for layer 2:'
nlayer = 2
nnf2 = 1 -- just one frames goes into layer 2
--nk2 = f2_
--nk = nk2
is = k2_
fanin = 2 -- createCoCnxTable creates also 2*fanin connections
feat_group = 32 --features per group (32=best in CIFAR, nk1=32, fanin=2)
poolsize = 2
cvstepsize = 1
ovhe2 = (ovhe-is+1)/poolsize/cvstepsize -- output video feature height
ovwi2 = (ovwi-is+1)/poolsize/cvstepsize -- output video feature width


-- OUTPUT Co-occurence CONNEX MODEL:
print '==> Computing connection tables based on co-occurence of features'
cTable2, kernels2 = createCoCnx(nlayer, videoData2, nk1, feat_group, fanin, 50, nnf2, is, kernels1, false)
nk2 = cTable2:max()
nk = nk2
if opt.display then image.display{image=kernels2:reshape(kernels2:size(1),is,is), 
		padding=2, symmetric=true, nrow = 32, zoom=2, legend = 'Layer 2 filters'} end


----------------------------------------------------------------------
-- 2nd layer

vnet2 = nn.Sequential()
vnet2:add(nn.SpatialConvolutionMap(cTable2, is, is, cvstepsize,cvstepsize)) -- connex table based on similarity of features
vnet2:add(nn.Threshold())
if opt.pooling == 'max' then
	vnet2:add(nn.SpatialMaxPooling(s1_,s1_,s1_,s1_))
elseif opt.pooling == 'sum' then
	vnet2:add(nn.SpatialSubSampling(f2_,s1_,s1_,s1_,s1_))
elseif opt.pooling == 'l2' then
	vnet2:add(nn.SpatialLPPooling(f2_,2,s1_,s1_,s1_,s1_))
end


-- setup net/ load kernels into network:
vnet2.modules[1].bias = vnet2.modules[1].bias*0 -- set bias to 0!!! not needed
kernels2_ = kernels2:clone():div(nk2/2) -- divide kernels so output of SpatialConv is about ~1 or more
vnet2.modules[1].weight = kernels2_  -- OR-AND model *3/2 because of fanin and 2*fanin connnex table

----------------------------------------------------------------------
print '==> process video throught 2nd layer:'
print 'Initial frames will be blank because of the VolConv on 1st layer~'

videoData3 = processLayer(nlayer, vnet2, videoData2, nk2, ovhe2, ovwi2)

--report some statistics:
print('2nd layer conv out. std: '..vnet2.modules[1].output:std()..' and mean: '..vnet2.modules[1].output:mean())
print('2nd layer output. std: '..vnet2.output:std()..' and mean: '..vnet2.output:mean())



----------------------------------------------------------------------
print '==> generating filters for layer 3:'
nlayer = 3
nnf3 = 1  -- just one frames goes into layer 3
--nk3 = f3_
is = k3_
fanin = 2
feat_group = 8 
cvstepsize = 1
--ovhe3 = (ovhe2-is+1)/poolsize/cvstepsize -- output video feature height
--ovwi3 = (ovwi2-is+1)/poolsize/cvstepsize -- output video feature width


-- OUTPUT Co-occurence CONNEX MODEL:
print '==> Computing connection tables based on co-occurence of features'
cTable3, kernels3 = createCoCnx(nlayer, videoData3, nk2, feat_group, fanin, 50, nnf3, is, kernels2, false)
nk3 = cTable3:max()
--nk = nk3
if opt.display then image.display{image=kernels3, padding=2, padding=2, symmetric=true, 
		nrow = 64, zoom=2, legend = 'Layer 3 filters'} end

   
----------------------------------------------------------------------
-- 3rd layer   

vnet3 = nn.Sequential()
vnet3:add(nn.SpatialConvolutionMap(cTable3, is, is, cvstepsize,cvstepsize)) -- connex table based on similarity of features


-- setup net/ load kernels into network:
vnet3.modules[1].bias = vnet3.modules[1].bias*0 -- set bias to 0!!! not needed
kernels3_ = kernels3:clone():div(nk3*fanin) -- divide kernels so output of SpatialConv std ~0.5
vnet3.modules[1].weight = kernels3_
   
   
---------------------------------------------------------------------- 
-- quick sanity check with Lena:

--normkernel = image.gaussian1D(15)
--normer=nn.SpatialContrastiveNormalization(3, normkernel,1e-3)
--lvn=normer:forward(image.lena())
--lv1 = vnet:forward(lvn)
--image.display(lv1)
--lv2 = vnet2:forward(lv1)
--image.display(lv2)


   
----------------------------------------------------------------------  
-- prepare full network with all layers:
tnet = vnet:clone()
for i=1,vnet2:size() do
   tnet:add(vnet2.modules[i]:clone())
end
for i=1,vnet3:size() do
   tnet:add(vnet3.modules[i]:clone())
end

convnet = tnet -- pointer to full convnet trained with CL

----------------------------------------------------------------------
--
   -- Replicate convnet:
   convnets = {}
   preprocessors = {}
   for i in ipairs(scales) do
      if opt.share == 'all' then
         table.insert(convnets, convnet:clone('weight','bias','gradWeight','gradBias'))
      elseif tonumber(opt.share) == 0 then
         table.insert(convnets, convnet:clone())
      elseif tonumber(opt.share) > 0 then
         opt.share = tonumber(opt.share)
         local newnet = convnet:clone()
         if opt.share >= 1 then
            newnet.modules[1].weight:set(convnet.modules[1].weight)
            newnet.modules[1].bias:set(convnet.modules[1].bias)
            newnet.modules[1].gradWeight:set(convnet.modules[1].gradWeight)
            newnet.modules[1].gradBias:set(convnet.modules[1].gradBias)
         end
         if opt.share >= 2 then
            newnet.modules[4].weight:set(convnet.modules[4].weight)
            newnet.modules[4].bias:set(convnet.modules[4].bias)
            newnet.modules[4].gradWeight:set(convnet.modules[4].gradWeight)
            newnet.modules[4].gradBias:set(convnet.modules[4].gradBias)
         end
         table.insert(convnets, newnet)
      end
      table.insert(preprocessors, preproc:clone())
   end

   -- Fovea Container
   fovea = nn.SpatialFovea{nInputPlane = 3,
                           ratios = scales,
                           preProcessors = preprocessors,
                           processors = convnets,
                           fov = trainable_fov,
                           sub = trainable_ss,
                           cachePrePreproc = true}

----------------------------------------------------------------------
--
   -- Classifier
   classifier = nn.Sequential()
   if opt.rbf then
      -- rbf classifier
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,p1_))
      classifier:add(nn.Tanh())
      classifier:add(nn.WeightedEuclidean(p1_,#classes))
      classifier:add(nn.Power(-2))
      --classifier:add(nn.Minus()) -- this does not exist...
   elseif p1_ == 0 then
      -- no hidden layers: a simple linear classifier
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,#classes))
   else
      -- a 2-layer perceptron
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,p1_))
      classifier:add(nn.Tanh())
      classifier:add(nn.Linear(p1_,#classes))
   end

   -- Global trainable machine
   trainable = classifier -- nn.Sequential() -- EC unsup network mods
   --trainable:add(fovea) --train only classifier with Clustering Learning/unsup network
   --trainable:add(classifier) --train only classifier with Clustering Learning/unsup network
   
   -- compute network creation time time 
   time = sys.clock() - time
   print("<net> time to CL train network = " .. (time*1000) .. 'ms')

   -- Save model
   if opt.save then
      print('<trainer> saving bare network to '..opt.save)
      os.execute('mkdir -p "' .. sys.dirname(opt.save) .. '"')
      torch.save(opt.save, trainable)
   end
--else
--   -- reload network
--   print('<trainer> reloading previously trained network')
--   trainable = torch.load(opt.network)
--
--   -- get pointers to internal modules
--   fovea = trainable.modules[1]
--   classifier = trainable.modules[2]
--   convnet = fovea.processors[1]
--
--   -- optionally replace classifier?
--   if p1_ ~= 0 and #classifier.modules < 5 then
--      -- replacing linear classifier by 2-layer perceptron
--      print('<trainer> dropping linear classifier, replacing by 2-layer perceptron')
--      classifier.modules[3] = nil
--      classifier:add(nn.Linear(p0_,p1_))
--      classifier:add(nn.Tanh())
--      classifier:add(nn.Linear(p1_,#classes))
--   end
--end

-- retrieve parameters and gradients
parameters,gradParameters = trainable:getParameters()

-- verbose
print('<trainer> model:')
print(trainable)
print('<trainer> nb of trainable parameters: ' .. parameters:size(1))

----------------------------------------------------------------------
-- training criterion: Mean-Square Error or Negative Log-Likelihood
--
criterion = nn.DistNLLCriterion()
criterion.targetIsProbability = true



----------------------------------------------------------------------
-- load/get dataset

--print '==> loading siftflow training-set'
--dofile('sf-dataset.lua')




----------------------------------------------------------------------
-- process images in dataset with unsupervised network 'tnet':
--

--trainsize = 200
--testsize = 200
trainsize = trainData:size()
testsize = testData:size()
--
--in_trsz = trainData[1][1]:size(2)
--in_tesz = testData[1][1]:size(2)
--if nnf1 >1 then
--   out_trsz = tnet:forward(trainData[1][1]:resize(ivch,1,in_trsz,in_trsz):expand(ivch,nnf1,in_trsz,in_trsz))
--   out_tesz = tnet:forward(testData[1][1]:resize(ivch,1,in_tesz,in_tesz):expand(ivch,nnf1,in_tesz,in_tesz))
--elseif nnf1 ==1 then
--   out_trsz = tnet:forward(trainData[1][1]:resize(ivch,in_trsz,in_trsz))
--   out_tesz = tnet:forward(testData[1][1]:resize(ivch,in_tesz,in_tesz))
--end




-------------------------
-- free up memory, reset fovea?

print "==> processing dataset with videoknet:"
trainData2 = {}
for t = 1,trainsize do
   trainData2[t] = {}
   local sample = trainData[t]
   local input = sample[1]
   if input:size(1) == 3 then
      local target = sample[2]
      local sample_x = sample[3]
      local sample_y = sample[4]
      local sample_size = sample[5]
      fovea:focus(sample_x, sample_y, sample_size)
      if nnf1>1 then trainData2[t][1] = fovea:forward(input:resize(ivch,1,in_trsz,in_trsz):expand(ivch,nnf1,in_trsz,in_trsz)):clone()
      elseif nnf1 ==1 then
         trainData2[t][1] = fovea:forward(input):clone()
      end
      trainData2[t][2] = target
   end 
   xlua.progress(t, trainsize)
end

--report some statistics:
print('trainData2[1][1] std: '..trainData2[1][1]:std()..' and mean: '..trainData2[1][1]:mean())


testData2 = {}
fovea:focus() -- defocus to process all dataset images in full size~
for t = 1, #testData.rawdata/5 do
   testData:loadSample((t-1)*5+1)
   testData2[t] = {}
   local input = testData.currentSample
   if nnf1>1 then testData2[t][1] = fovea:forward(input:resize(ivch,1,in_tesz,in_tesz)):clone()--:expand(ivch,nnf1,in_tesz,in_tesz))
   elseif nnf1 ==1 then testData2[t][1] = fovea:forward(input):clone()
   end
    testData2[t][2] = testData.currentMask:clone()
   xlua.progress(t, #testData.rawdata/5)
end

--report some statistics:
print('testData2[1][1] std: '..testData2[1][1]:std()..' and mean: '..testData2[1][1]:mean())




-------------------------
-- do training:
--dofile('sf-vkn-train.lua')


