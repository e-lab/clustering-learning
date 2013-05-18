----------------------------------------------------------------------
-- Clustering learning on CIFAR10 dataset 
-- May 2013
--
-- bypass route color version: pass color info to final classifier AND net input!
--
-- Author: Eugenio Culurciello, May 2013
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
--require 'slac'


-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>> Clustering learning on CIFAR10 dataset <<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

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

	--seed				      (default 1)          use fixed seed for randomized initialization
	--initstd               (default 0.1)        initial std for k-means
	--niter                 (default 15)         iterations for k-means
	--kmbatchsize           (default 1000)       batch size for k-means
	--numlayers             (default 3)          number of layers in network	
	
	--display			display training/testing samples while training
	--plot 				plot error/accuracy live (if false, still logged in a file)
	--log					log the whole session to a file
	--quicktest       true = small test, false = full code running
	--videodata       true = load video file, otherwise ??? data
	--cnnmodel        true = convnet model with tanh and normalization, otherwise without
   --slacmodel       slac model for multi-layer
]]

if opt.quicktest then opt.nsamples = 300 else opt.nsamples = 10000 end  -- patch samples to use

-- set trues:
opt.plot = true 
opt.cnnmodel = true 		
opt.colorbypass = true


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
is0,is1,is2,is3 = 7,5,3,3 	-- size of kernels
ss1,ss2   		 = 2,2 			-- size of subsamplers (strides)
scales          = 1 				-- scales
fanin 			 = 3 				-- createCoCnxTable creates also 2*fanin connections
feat_group 		 = 16 			--features per group (32=best in CIFAR nk1=32, fanin=2)
opt.hiddens 	 = 256 			-- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2 	 = nk3, opt.hiddens -- dimensions for top perceptron
classes 			 = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                     'horse', 'ship', 'truck'} -- classes of objects to find

normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data_cifar.lua'

-- input image dateaset params:
ivch = trainData.data[1]:size(1) -- channels
ivhe = trainData.data[1]:size(2) -- height
ivwi = trainData.data[1]:size(3) -- width

----------------------------------------------------------------------
print '==> generating CL unsupervised network:'

-- compute network CL train time
time = sys.clock()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 1
nk = nk1
ovhe = (ivhe-is1+1)/ss1 -- output video feature height
ovwi = (ivwi-is1+1)/ss1 -- output video feature width


if opt.slacmodel then -- leanr nk1*N filters, then narrow down to nk1:
   kernels1u, counts1 = trainLayer(nlayer, trainData.data, opt.nsamples, nil, 2*nk1, nnf1, is1, false)
   kernels1s, cTable1 = slac(kernels1u, 2*nk1, nk1, 4, 4.5) -- SLAC algorithm to aggregate kernels
   nk1s=kernels1s:size(1)
   if opt.display then image.display{image=kernels1s:reshape(kernels1s:size(1),ivch,is1,is1), padding=2, symmetric=true, zoom=4, legend = 'Layer 1 filters'} end
else 
   -- AND/OR model or FULL CONNECT MODEL:
   -- get twice the kernels, then pick best ones:
   kernels1u, counts1 = trainLayer(nlayer, trainData.data, opt.nsamples, nil, 2*nk1, nnf1, is1, false)
   -- sort kernels:
   _, j = torch.sort(counts,true)
   kernels1 = torch.Tensor(nk1,ivch, is1,is1)
   for i=1,nk1 do
      kernels1[i] = kernels1u[j[i]]
   end
   if opt.display then
      image.display{image=kernels1:reshape(kernels1:size(1),ivch,is1,is1), padding=2, symmetric=true, zoom=4, legend = 'Layer 1 filters'} end
end

   
----------------------------------------------------------------------
-- 1st layer
  
-- Trainable Network
vnet = nn.Sequential()
if opt.slacmodel then 
   vnet:add(nn.SpatialConvolution(ivch, nk1s, is1, is1))
   vnet:add(nn.SpatialMaxMap(cTable1))
else
   vnet:add(nn.SpatialConvolution(ivch, nk1, is1, is1))
end
vnet:add(nn.Threshold())
vnet:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))


-- setup net/ load kernels into network:
vnet.modules[1].bias = vnet.modules[1].bias*0 -- set bias to 0!!! not needed
if opt.slacmodel then 
   kernels1_ = kernels1s:clone():div(nnf1*nk1) -- divide kernels so output of SpatialConv std =~0.5
   vnet.modules[1].weight = kernels1_:reshape(nk1s, ivch, i1s,is1)
else 
   kernels1_ = kernels1:clone():div(nnf1*nk1) -- divide kernels so output of SpatialConv std =~0.5
   vnet.modules[1].weight = kernels1_:reshape(nk1, ivch, is1,is1)
end


----------------------------------------------------------------------
print '==> process dataset throught 1st layer:'
trainData2, stdc1, meac1, stdo, meao = processLayer(nlayer, vnet, trainData.data, nk1, ovhe, ovwi, false)
testData2 = processLayer(nlayer, vnet, testData.data, nk1, ovhe, ovwi, false)

--report some statistics:
print('1st layer conv out. std: '..stdc1..' and mean: '..meac1)
print('1st layer output. std: '..stdo..' and mean: '..meao)


----------------------------------------------------------------------
print '==> generating filters for layer 2:'
nlayer = 2
nk = nk2
ovhe2 = (ovhe-is2+1)/ss2 -- output video feature height
ovwi2 = (ovwi-is2+1)/ss2 -- output video feature width

--if opt.slacmodel then
--   -- SLAC MODEL: nk1*N filters to learn, then narrow down to nk1:
--   nk22 = nk2*2
--   kernels2u, counts2 = trainLayer(nlayer, trainData2, opt.nsamples, nil, nk22, nnf2, is2, false)
--   kernels2s, cTable2 = slac(kernels2u, nk22, nk2, 4, 4.5) -- SLAC algorithm to aggregate kernels
--else 

-- OUTPUT Co-occurence CONNEX MODEL:
print '==> Computing connection tables based on co-occurence of features'
cTable2, kernels2 = createCoCnx(nlayer, trainData2[{{1,1024}}], nk1, feat_group, fanin, opt.nsamples, nnf2, is2, false)
nk2 = cTable2:max()
nk = nk2
if opt.display then image.display{image=kernels2:reshape(kernels2:size(1),is2,is2), padding=2, 
      symmetric=true, nrow = 32, zoom=4, legend = 'Layer 2 filters'} end


----------------------------------------------------------------------
-- 2nd layer
vnet2 = nn.Sequential()
vnet2:add(nn.SpatialConvolutionMap(cTable2, is2, is2, cvstepsize,cvstepsize)) -- connex table based on similarity of features
vnet2:add(nn.Threshold())
vnet2:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))

-- setup net/ load kernels into network:
vnet2.modules[1].bias = vnet2.modules[1].bias*0 -- set bias to 0!!! not needed
kernels2_= kernels2:clone():div(15) -- divide kernels so output of SpatialConv std =~0.5
vnet2.modules[1].weight = kernels2_:reshape(kernels2_:size(1),is2,is2)  -- OR-AND model *3/2 because of fanin and 2*fanin connnex table

----------------------------------------------------------------------
print '==> process dataset throught 2nd layer:'

trainData3, stdc1, meac1, stdo, meao = processLayer(nlayer, vnet2, trainData2, nk2, ovhe2, ovwi2, false)
testData3 = processLayer(nlayer, vnet2, testData2, nk2, ovhe2, ovwi2, false)

--report some statistics:
print('2nd layer conv out. std: '..stdc1..' and mean: '..meac1)
print('2nd layer output. std: '..stdo..' and mean: '..meao)


----------------------------------------------------------------------
-- compute network creation time time 
time = sys.clock() - time
print("<net> time to CL train network = " .. (time*1000) .. 'ms')



----------------------------------------------------------------------
-- process images in dataset with unsupervised network 'tnet':
-- 

print "==> processing dataset with videoknet:"
-- train:
trainData2 = {}
trainData2.data = trainData3
trainData2.labels = trainData.labels
-- test:
testData2 = {}
testData2.data = testData3
testData2.labels = testData.labels

if not opt.colorbypass then -- then this is the final dataset!
	trainData.data = trainData2
	testData.data = testData2
end
--report some statistics:
print('testData.data[1] std: '..testData.data[1]:std()..' and mean: '..testData.data[1]:mean())
print('trainData.data[1] std: '..trainData.data[1]:std()..' and mean: '..trainData.data[1]:mean())


----------------------------------------------------------------------
-- Color bypass
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

	for i=1,100 do
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


