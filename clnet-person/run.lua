----------------------------------------------------------------------
-- Test of clustering learning on INRIA pedestrian, etc datasets
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
require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require 'online-kmeans' -- allow you to re-train k-means kernels
--require 'unsup' -- standard kmeans
--require 'topo-kmeans' -- new topographic disc kmeans (gives about same results...)

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
	--seed				(default 1)					use fixed seed for randomized initialization
]]

opt.initstd = 0.1
opt.niter = 15
opt.kmbatchsize = 1000 -- kmeans batchsize

opt.plot = false -- because otherwise it would be a string...
opt.quicktest = false	--(default 0)			true = small test, false = full code running
opt.cnnmodel = true --(default 1)			true = convnet model with tanh and normalization, otherwise without
opt.videodata = false --	(default 1) 		true = load video file, otherwise ??? data
opt.colorbypass = true
opt.numlayers = 1 -- number of layers in network

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

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
-- define network to train
--
print('<trainer> creating new network')

nnf1,nnf2,nnf3  = 1,1,1 		-- number of frames at each layer
nk0,nk1,nk2,nk3 = 3,32,64,128 -- nb of features
is0,is1,is2,is3 = 15,7,7,7 	-- size of kernels
ss1,ss2   		 = 2,2 			-- size of subsamplers (strides)
scales          = 1 				-- scales
fanin 			 = 2 				-- createCoCnxTable creates also 2*fanin connections
feat_group 		 = 16 			--features per group (16=best in INRIA nk1=32, fanin=2)
opt.hiddens 	 = 128 			-- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2 	 = nk3, opt.hiddens -- dimensions for top perceptron
classes 			 = {'person', 'bg'} -- classes of objects to find

normkernel = image.gaussian1D(7)

-- Preprocessor (normalizer)
normthres = 1e-1
preproc = nn.Sequential()
preproc:add(nn.SpatialColorTransform('rgb2yuv'))
do
	ynormer = nn.Sequential()
	ynormer:add(nn.Narrow(1,1,1))
	ynormer:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(is0), normthres))
	normer = nn.ConcatTable()
	normer:add(ynormer)
	normer:add(nn.Narrow(1,2,2))
end
preproc:add(normer)
preproc:add(nn.JoinTable(1))



----------------------------------------------------------------------
if opt.videodata then
   print '==> loading videt training-set:'
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
   print '==> loading the INRIA Person dataset:'
	data  = require 'data-person'
   
   -- input image dateaset params:
   ivch = trainData.data[1]:size(1) -- channels
   ivhe = trainData.data[1]:size(2) -- height
   ivwi = trainData.data[1]:size(3) -- width
   
end

-- number of frames to process:
if opt.quicktest then nfpr = 10 -- batch process size [video frames]
else nfpr = 200 end


----------------------------------------------------------------------
-- normalize input video data:

function createDataBatch()
   videoData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
   if opt.videodata then
   	for i = 1, nfpr do -- just get a few frames from a video as dataset:
      	-- perform full LCN
      	procFrame = preproc:forward(rawFrame) 
      end
      videoData[i] = procFrame
      rawFrame = source:forward()
   else 
   	videoData = trainData.data[{{1,nfpr}}] 
   end
   
   return videoData
end

createDataBatch()

if opt.display then image.display{image=videoData[{{1,9}}],
		padding=2, symmetric=true, nrow = 3, zoom=1, legend = 'Input Video Data [normalized]'} end


----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> generating CL unsupervised network:'

-- compute network CL train time
   time = sys.clock()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 1
nnf1 = 1 -- number of frames from input video to use
if opt.quicktest then opt.nsamples = 300 else opt.nsamples = 10000 end  -- patch samples to use
ovhe = (ivhe-is1+1)/ss1 -- output video feature height
ovwi = (ivwi-is1+1)/ss1 -- output video feature width

 
-- AND/OR model or FULL CONNECT MODEL:
kernels1 = trainLayer(nlayer, videoData, opt.nsamples, nil, nk1, nnf1, is1, true) 

if opt.display then image.display{image=kernels1:reshape(nk1,ivch,is1,is1),
		padding=2, symmetric=true, nrow = 8, zoom=2, legend = 'Layer 1 filters'} end

   
----------------------------------------------------------------------
-- 1st layer

vnet = nn.Sequential()
if nnf1 > 1 then
	vnet:add(nn.VolumetricConvolution(ivch, nk1, nnf1, is1, is1, 1))
	vnet:add(nn.Sum(2)) -- needed by volconv
elseif nnf1 == 1 then
	vnet:add(nn.SpatialConvolution(ivch, nk1, is1, is1))
end
vnet:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
vnet:add(nn.Threshold())

-- setup net/ load kernels into network:
vnet.modules[1].bias = vnet.modules[1].bias*0 -- set bias to 0!!! not needed
kernels1_ = kernels1:clone():div(nnf1*nk1) -- divide kernels so output of SpatialConv std ~0.5
vnet.modules[1].weight = kernels1_:reshape(nk1,ivch,is1,is1)


----------------------------------------------------------------------
print '==> process video throught 1st layer:'
videoData2, stdc1, meac1, stdo, meao  = processLayer(nlayer, vnet, videoData, nk1, ovhe, ovwi)
videoData = nil -- free space!

--report some statistics:
print('1st layer conv out. std: '..stdc1..' and mean: '..meac1)
print('1st layer output. std: '..stdo..' and mean: '..meao)


if opt.numlayers >=2 then
	----------------------------------------------------------------------
	print '==> generating filters for layer 2:'
	nlayer = 2
	ovhe2 = (ovhe-is2+1)/ss2 -- output video feature height
	ovwi2 = (ovwi-is2+1)/ss2 -- output video feature width


	-- OUTPUT Co-occurence CONNEX MODEL:
	print '==> Computing connection tables based on co-occurence of features: [nk1*feat_group*(fanin+fanin*2)]'
	cTable2, kernels2 = createCoCnx(nlayer, videoData2, nk1, feat_group, fanin, opt.nsamples/10, nnf2, is2, false)
	nk2 = cTable2:max()
	nk = nk2
	if opt.display then image.display{image=kernels2:reshape(kernels2:size(1),is2,is2), 
			padding=2, symmetric=true, nrow = 64, zoom=2, legend = 'Layer 2 filters'} end


	----------------------------------------------------------------------
	-- 2nd layer

	vnet2 = nn.Sequential()
	vnet2:add(nn.SpatialConvolutionMap(cTable2, is2, is2)) -- connex table based on similarity of features
	vnet2:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))
	vnet2:add(nn.Threshold())

	-- setup net/ load kernels into network:
	vnet2.modules[1].bias = vnet2.modules[1].bias*0 -- set bias to 0!!! not needed
	kernels2_ = kernels2:clone():div(nk2/3) -- divide kernels so output of SpatialConv std is ~0.5
	vnet2.modules[1].weight = kernels2_  -- OR-AND model *3/2 because of fanin and 2*fanin connnex table

	----------------------------------------------------------------------
	print '==> process video throught 2nd layer:'
	videoData3, stdc1, meac1, stdo, meao = processLayer(nlayer, vnet2, videoData2, nk2, ovhe2, ovwi2)
	videoData2 = nil -- free space!

	--report some statistics:
	print('2nd layer conv out. std: '..stdc1..' and mean: '..meac1)
	print('2nd layer output. std: '..stdo..' and mean: '..meao)
end  -- opt.numlayer >=2 

if opt.numlayers >=3 then
	----------------------------------------------------------------------
	print '==> generating filters for layer 3:'
	nlayer = 3
	nnf3 = 1  -- just one frames goes into layer 3
	feat_group = 8 
	cvstepsize = 1
	ovhe3 = (ovhe2-is3+1) -- output video feature height
	ovwi3 = (ovwi2-is3+1) -- output video feature width


	-- OUTPUT Co-occurence CONNEX MODEL:
	print '==> Computing connection tables based on co-occurence of features'
	cTable3, kernels3 = createCoCnx(nlayer, videoData3, nk2, feat_group, fanin, opt.nsamples/10, nnf3, is3, false)
	nk3 = cTable3:max()
	--nk = nk3
	if opt.display then image.display{image=kernels3, padding=2, padding=2, symmetric=true, 
			nrow = 64, zoom=2, legend = 'Layer 3 filters'} end

	
	----------------------------------------------------------------------
	-- 3rd layer   

	vnet3 = nn.Sequential()
	vnet3:add(nn.SpatialConvolutionMap(cTable3, is3, is3)) -- connex table based on similarity of features


	-- setup net/ load kernels into network:
	vnet3.modules[1].bias = vnet3.modules[1].bias*0 -- set bias to 0!!! not needed
	kernels3_ = kernels3:clone():div(nk3*2) -- divide kernels so output of SpatialConv std ~0.5
	vnet3.modules[1].weight = kernels3_

	----------------------------------------------------------------------
	print '==> process video throught 3rd layer:'
	videoData4, stdc1, meac1, stdo, meao = processLayer(nlayer, vnet3, videoData3, nk3, ovhe3, ovwi3) -- just a few samples
	videoData3 = nil -- free space!

	--report some statistics:
	print('3rd layer conv out. std: '..stdc1..' and mean: '..meac1)
	print('3rd layer output. std: '..stdo..' and mean: '..meao)
end  -- opt.numlayer >=3    
   
---------------------------------------------------------------------- 

-- compute network creation time time 
time = sys.clock() - time
print("==>  time to CL train network = " .. (time*1000) .. 'ms')

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
-- prepare full network [tnet] with all layers:
tnet = vnet:clone()
if opt.numlayers >=2 then
	for i=1,vnet2:size() do
		tnet:add(vnet2.modules[i]:clone())
	end
end
if opt.numlayers >=3 then
	for i=1,vnet3:size() do
		tnet:add(vnet3.modules[i]:clone())
	end
end



----------------------------------------------------------------------
-- process images in dataset with unsupervised network 'tnet':
-- 

print "==> loading dataset:"
if not data then data  = require 'data-person' end


print "==> processing dataset with videoknet:"
-- train:
tfa = #tnet:forward(trainData.data[1])
trainData2 = {}
trainData2.data = torch.Tensor(trainData:size(), tfa[1], tfa[2], tfa[3])
for i = 1,trainData:size() do
	trainData2.data[i] = tnet:forward(trainData.data[i])
	xlua.progress(i, trainData:size())
end
-- test:
testData2 = {}
testData2.data = torch.Tensor(testData:size(), tfa[1], tfa[2], tfa[3])
for i = 1,testData:size() do
	testData2.data[i] = tnet:forward(testData.data[i])
	xlua.progress(i, testData:size())
end

if not opt.colorbypass then -- then this is the final dataset!
	trainData.data = trainData2
	testData.data = testData2
end
--report some statistics:
print('testData.data[1] std: '..testData.data[1]:std()..' and mean: '..testData.data[1]:mean())
print('trainData.data[1] std: '..trainData.data[1]:std()..' and mean: '..trainData.data[1]:mean())


----------------------------------------------------------------------
-- Color bypass
--if opt.colorbypass then
--	totalpool = ss1*ss2
--	trainData, testData = colorBypass(totalpool, trainData2 , testData2) -- will operate on trainData2 , testData2 	
--	cl_nk1 = (#trainData.data)[2] -- resize output of the concatenated vector
--end
--
--
if opt.colorbypass then
	totalpool = ss1 -- only 1 layer used!
	if opt.numlayers >= 2 then totalpool = totalpool*ss2 end -- in case we use 2,3 layers!
	trainData, testData = colorBypass(totalpool, trainData2 , testData2) -- will operate on trainData2 , testData2
	cl_nk1 = (#trainData.data)[2] * (#trainData.data)[3] * (#trainData.data)[4] 
end

----------------------------------------------------------------------
-- Classifier

-- functions for Clustering Learning classifier:
dofile('clclassifier.lua')

-- standard mlp:
if true then
	-- MLP classifier:
	model = nn.Sequential()
	-- a 2-layer perceptron
	model:add(nn.Tanh())
	model:add(nn.Reshape(cl_nk1))
	model:add(nn.Linear(cl_nk1,cl_nk2))
	model:add(nn.Tanh())
	model:add(nn.Linear(cl_nk2,#classes))
	-- final stage: log probabilities
	model:add(nn.LogSoftMax())

	-- Loss: NLL
	loss = nn.ClassNLLCriterion()

	-- verbose
	print('==>  model:')
	print(model)

	----------------------------------------------------------------------
	-- load/get dataset
	print '==> load modules'

	train = require 'train'
	test  = require 'test'

	----------------------------------------------------------------------
	print '==> training!'

	while true do
		train(trainData)
		test(testData)
	end
	
	-- Save model for demos:
	if opt.save then
	
		-- replace classifier (2nd module) by SpatialClassifier
		sclassifier = nn.SpatialClassifier(model)
		tnet:add(sclassifier)	
		
		print('==>  <trainer> saving bare network to '..opt.save)
		os.execute('mkdir -p "' .. sys.dirname(opt.save) .. '"')
		torch.save(opt.save..'demo.net', tnet)
	end

else
	----------------------------------------------------------------------
	-- DISTANCE CL Classifier:

	-- train clusters on each trainData category separately:
	results = {}--torch.Tensor(20,2)
	nclusters = 32 -- number of clusters per class
	i=1
	for fracDataset = 0.1, 1, 0.1 do
		clusteredClasses = trainCLClassifier(fracDataset,nclusters)
		-- test on train and test sets:
		ctr, cte = testCLnet(fracDataset, clusteredClasses, nclusters)
		table.insert(results, {ctr, cte})
		--results[i]=torch.Tensor({ctr, cte})
		i = i+1
	end

	require 'csv'
	csv.save('multinet_results.txt', results)
	
	--------

	-- image of features:
	imaFeats = torch.Tensor(128,128)--trainData.data:size(1), trainData.data:size(2))
	j=1
	for i = 1, 256 do --trainData:size() do
		if trainData.labels[i] == 1 then
			imaFeats[j] = trainData.data[j]:clone():reshape(128)
			j = j+1
		end
		--xlua.progress(i, trainData:size())
	end
	image.display{image=imaFeats, zoom=4}--, symmetric=true}

end

