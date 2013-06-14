----------------------------------------------------------------------
-- This script loads the SVHN dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet, E. Culurciello
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset


www = 'http://data.neuflow.org/data/housenumbers/'

train_file = '../../datasets//housenumbers/train_32x32.t7'
test_file = '../../datasets//housenumbers/test_32x32.t7'
extra_file = '../../datasets//housenumbers/extra_32x32.t7'


if not paths.filep(train_file) then
   os.execute('wget ' .. www .. train_file)
end
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end
if opt.size == 'extra' and not paths.filep(extra_file) then
   os.execute('wget ' .. www .. extra_file)   
end

----------------------------------------------------------------------
-- training/test size

trsize = 73257
tesize = 26032

if opt.size == 'extra' then
   print '==> using extra training data'
   trsize = 73257 + 531131
   tesize = 26032
elseif opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 73257
   tesize = 26032
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 10000
   tesize = 2000
end

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

if opt.size == 'extra' then
   loaded = torch.load(extra_file,'ascii')
   trdata = torch.Tensor(trsize,3,32,32)
   trdata[{ {1,(#trainData.data)[1]} }] = trainData.data
   trdata[{ {(#trainData.data)[1]+1,-1} }] = loaded.X:transpose(3,4)
   trlabels = torch.Tensor(trsize)
   trlabels[{ {1,(#trainData.labels)[1]} }] = trainData.labels
   trlabels[{ {(#trainData.labels)[1]+1,-1} }] = loaded.y[1]
   trainData = {
      data = trdata,
      labels = trlabels,
      size = function() return trsize end
   }
end

-- Finally we load the test data.

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   first256Samples_r = trainData.data[{ {1,256},1 }]
   image.display{image=first256Samples_r, nrow=16, legend='Some training examples: ' ..channels[1].. ' channel'}
end
