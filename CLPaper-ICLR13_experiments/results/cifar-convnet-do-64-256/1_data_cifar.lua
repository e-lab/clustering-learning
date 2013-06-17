----------------------------------------------------------------------
-- This script loads the CIFAR10 dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet, E. Culurciello
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('CIFAR Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
--   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> downloading dataset'

tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'

if not paths.dirp('../datasets/cifar-10-batches-t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

----------------------------------------------------------------------
print '==> loading dataset'

-- dataset size:
trsize = 50000
tesize = 2000

trainData = {
   data = torch.Tensor(trsize, 3*32*32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('../datasets/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('../datasets/cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trsize = 20000  -- repeated here for smaller size train/test
tesize = 2000

trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

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
-- EC: removed not bio-inspired!
--print '==> preprocessing data: colorspace RGB -> YUV'
--for i = 1,trsize do
--   trainData.data[i] = image.rgb2yuv(trainData.data[i])
--end
--for i = 1,tesize do
--   testData.data[i] = image.rgb2yuv(testData.data[i])
--end

-- Name channels for convenience
--channels = {'y','u','v'}
channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
--print '==> preprocessing data: normalize each feature (channel) globally'
--mean = {}
--std = {}
--for i,channel in ipairs(channels) do
--   -- normalize each channel globally:
--   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
--   std[i] = trainData.data[{ {},i,{},{} }]:std()
--   trainData.data[{ {},i,{},{} }]:add(-mean[i])
--   trainData.data[{ {},i,{},{} }]:div(std[i])
--end
--
---- Normalize test data, using the training means/stds
--for i,channel in ipairs(channels) do
--   -- normalize each channel globally:
--   testData.data[{ {},i,{},{} }]:add(-mean[i])
--   testData.data[{ {},i,{},{} }]:div(std[i])
--end

-- Local normalization
-- (note: the global normalization is useless, if this local normalization
-- is applied on all channels... the global normalization code is kept just
-- for the tutorial's purpose)
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
--if not is then is = 7 end -- find is value from call-out script
--print("Normalizing kernel size is:", is)
neighborhood = image.gaussian1D(9)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trsize do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,tesize do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

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
   first256Samples_y = trainData.data[{ {1,256},1 }]
   first256Samples_u = trainData.data[{ {1,256},2 }]
   first256Samples_v = trainData.data[{ {1,256},3 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: ' ..channels[1].. ' channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: ' ..channels[2].. ' channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: ' ..channels[3].. ' channel'}
end
