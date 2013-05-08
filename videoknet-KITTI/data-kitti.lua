----------------------------------------------------------------------
-- This script loads the KITTI dataset
-- training data, and pre-process it to facilitate learning.
-- Aysegul Dundar
-- Date : 04/28/2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require 'sys'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('KITTI Dataset Preprocessing')
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

if (paths.filep('train.t7') 
		and paths.filep('test.t7')) then

	print '==> loading previously generated dataset:'
	trainData = torch.load('train.t7')
	testData = torch.load('test.t7')

else 

    dofile 'createKITTIdataset.lua'
	
end

trsize = trainData.data:size(1)
tesize = testData.data:size(1)
print('trsize', trsize)


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
channels = {'r', 'g', 'b'}
-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: global normalization:'
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

for i = 1,trsize do
   trainData.data[i] = preproc:forward(trainData.data[i])
end
for i = 1,tesize do
   testData.data[i] = preproc:forward(testData.data[i])
end


----------------------------------------------------------------------
print '==> verify statistics:'

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
print '==> visualizing data:'

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


-- Exports
return {
   trainData = trainData,
   testData = testData
}
