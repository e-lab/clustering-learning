----------------------------------------------------------------------
-- This script loads the CIFAR10 dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet, E. Culurciello
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

----------------------------------------------------------------------

tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'
if not paths.dirp('../../datasets/cifar-10-batches-t7') then
   print '==> downloading dataset'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

-- dataset size: -- will be resized below by opt.smalldata!!!!! be cautious!
trsize = 50000
tesize = 10000

trainData = {
   data = torch.Tensor(trsize, 3*32*32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 0,4 do
   subset = torch.load('../../datasets//cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():float()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels:float()
end
trainData.labels = trainData.labels + 1

subset = torch.load('../../datasets//cifar-10-batches-t7/test_batch.t7', 'ascii')
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

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)


----------------------------------------------------------------------
channels = {'r','g','b'}

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
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   print '==> visualizing data'
   local first256Samples_r = trainData.data[{ {1,256},1 }]
   image.display{image=first256Samples_r, nrow=16, legend='Some training examples: ' ..channels[1].. ' channel'}
end
