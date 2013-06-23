----------------------------------------------------------------------
-- This script loads the CIFAR10 dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet, E. Culurciello
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

----------------------------------------------------------------------

local tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'
if not paths.dirp('../../datasets/cifar-10-batches-t7') then
   print '==> downloading dataset'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

-- dataset size: -- will be resized below by opt.smalldata!!!!! be cautious!
local trsize = 50000
local tesize = 10000

trainData = {
   data = torch.Tensor(trsize, 3,32,32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 0,4 do
   local subset = torch.load('../../datasets//cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():float()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels:float()
end
trainData.labels = trainData.labels + 1

local subset = torch.load('../../datasets//cifar-10-batches-t7/test_batch.t7', 'ascii')

testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- dataset size:
if opt.size=='small' then
	trsize = 10000
	tesize = 2000

    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]
    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]
end

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
--print '==> preprocessing data: colorspace RGB -> YUV'
--for i = 1,trainData:size() do
--   trainData.data[i] = image.rgb2yuv(trainData.data[i])
--end
--for i = 1,testData:size() do
--   testData.data[i] = image.rgb2yuv(testData.data[i])
--end


-- Name channels for convenience
local channels = {'r','g','b'}

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
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: U channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: V channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std
}

