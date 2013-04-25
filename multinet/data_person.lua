----------------------------------------------------------------------
-- This script loads the INRIA person dataset
-- training data, and pre-process it to facilitate learning.
-- E. Culurciello
-- April 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require 'unsup'
require 'sys'
require 'ffmpeg'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('INRIA Person Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
--   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- 

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!


----------------------------------------------------------------------
-- load or generate new dataset:

if paths.filep('../../datasets/INRIAPerson/train.t7') 
		and paths.filep('../../datasets/INRIAPerson/test.t7') then

	print '==> loading previously generated dataset:'
	trainData = torch.load('../../datasets/INRIAPerson/train.t7')
	testData = torch.load('../../datasets/INRIAPerson/test.t7')
	
	trsize = trainData.data:size(1)
	tesize = testData.data:size(1)

else

	print '==> creating a new dataset from raw files:'

	-- video dataset to get background from:
	dspath = '../../datasets/driving1.mov'
	source = ffmpeg.Video{path=dspath, width = 160, height = 120, encoding='jpg', 
			fps=24, loaddump=false, load=false}

	rawFrame = source:forward()
	-- input video params:
	--ivch = rawFrame:size(1) -- channels
	ivhe = rawFrame:size(2) -- height
	ivwi = rawFrame:size(3) -- width
	source.current = 1 -- rewind video frames

	ivch = 3 -- color channels in images
	desImaX = 32 -- desired cropped dataset image size
	desImaY = 32

	cropTrX = 50 -- desired offset to crop images from train set
	cropTrY = 45
	cropTeX = 33 -- desired offset to crop images from test set
	cropTeY = 30

	labelPerson = 1 -- label for person and background:
	labelBg = 0

	trainDir='../../datasets/INRIAPerson/96X160H96/Train/pos/'
	trainImaNumber = #ls(traindir)
	testDir='../../datasets/INRIAPerson/70X134H96/Test/pos/'
	testImaNumber = #ls(testdir)

	-- dataset size:
	dataMultiplier = 1 -- optional: take multiple samples per image: +/- 2 pix H, V = 4 total
	trsize = dataMultiplier * trainImaNumber
	tesize = dataMultiplier * testImaNumber

	trainData = {
		data = torch.Tensor(trsize, ivch,desImaX,desImaY),
		labels = torch.Tensor(trsize),
		size = function() return trsize end
	}

	-- load person data:
	for i = 1, train_ima_number, 2 do
		imatoload = image.loadPNG(traindir..ls(trainDir)[i],ivch)
		trainData.data[i] = image.crop(imatoload, cropTrX-desImaX/2, cropTrY-desImaY/2, 
																cropTrX+desImaX/2, cropTrY+desImaY/2):clone()
		trainData.labels[i] = label_person
	
		-- load background data:
		imatoload = source:forward()
		local x = math.random(1,ivwi-desImaX+1)
		local y = math.random(20,ivhe-desImaY+1-40) -- added # to get samples more or less from horizon
		trainData.data[i+1] = imatoload[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
		trainData.labels[i+1] = label_bg
	end

	image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}


	testData = {
		data = torch.Tensor(tesize, ivch,desImaX,desImaY),
		labels = torch.Tensor(tesize),
		size = function() return tesize end
	}

	-- load person data:
	for i = 1, test_ima_number, 2 do
		imatoload = image.loadPNG(testdir..ls(testDir)[i],ivch)
		testData.data[i] = image.crop(imatoload, cropTeX-desImaX/2, cropTeY-desImaY/2, 
																cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
		testData.labels[i] = label_person
	
		-- load background data:
		imatoload = source:forward()
		local x = math.random(1,ivwi-desImaX+1)
		local y = math.random(20,ivhe-desImaY+1-40) -- added # to get samples more or less from horizon
		testData.data[i+1] = imatoload[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
		testData.labels[i+1] = label_bg
	end

	image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

	--save if needed:
	--torch.save('train.t7',trainData)
	--torch.save('test.t7',testData)
end


print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trsize do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,tesize do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}
--channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
-- (note: the global normalization is useless, if this local normalization
-- is applied on all channels... the global normalization code is kept just
-- for the tutorial's purpose)
--print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(7)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()

-- Normalize all channels locally:
--for c in ipairs(channels) do
c = 1
   for i = 1,trsize do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,tesize do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
--end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   local first256Samples_u = trainData.data[{ {1,256},2 }]
   local first256Samples_v = trainData.data[{ {1,256},3 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: ' ..channels[1].. ' channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: ' ..channels[2].. ' channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: ' ..channels[3].. ' channel'}
end
