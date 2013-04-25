----------------------------------------------------------------------
--Cosmo Zhang Spring 2013
--nn
--Traffic Sign
----------------------------------------------------------------------

import 'torch'
require 'csv'
require 'image'
require 'unsup'
require 'nnx'
--require 'eex'
require 'nn'
-- parse command line arguments                                                                                                                   
if not opt then                                                                                                                                     print '==> processing options'                                                                                                                   cmd = torch.CmdLine()                                                                                                                            cmd:text()                                                                                                                                       cmd:text('SVHN Dataset Preprocessing')                                                                                                           cmd:text()                                                                                                                                       cmd:text('Options:')                                                                                                                             cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')                                                                cmd:option('-visualize', true, 'visualize input data and weights during training')                                                               cmd:text()                                                                                                                                       opt = cmd:parse(arg or {})                                                                                                                    end 
function sleep(n)
   os.execute("sleep " .. tonumber(n))
end

if not qt then
--   opt.visualize = false
end
----------------------------------
--[[
   torch.manualSeed(opt.seed)
   torch.setnumthreads(opt.threads)
   torch.setdefaulttensortype('torch.FloatTensor')

   is = opt.inputsize
   nk1 = opt.nkernels1
   nk2 = opt.nkernels2
   nk3 = opt.nkernels3
]]
normkernel = image.gaussian1D(5)

----------------------------------------------------------------------
--print '==> downloading dataset'


if not sys.dirp(opt.datafile) then
   print('Error! Path wrong')
end

----------------------------------------------------------------------
-- training/test size

----------------------------------------------------------------------
print '==> loading dataset'

Dataload = nn.DataList()
a={0, 1, 3, 4, 5, 14, 26, 27, 29, 33}
--a={0, 1}
--for i=0, 5 do
for i, v in ipairs(a) do
   if v <10 then
      fname = '0000'..v
      cl='0'..v
   else
      fname = '000'..v
      cl=''..v
   end

   data = nn.DataSet{dataSetFolder=opt.datafile..fname, 
		     cacheFile=opt.datafile..fname,
		     nbSamplesRequired=opt.patches,
		     channels=3}
   --channels=1}
   --print(type(cl))
   --print(data[1][2])
   data:shuffle()
   Dataload:appendDataSet(data, cl)
end


Dataload:shuffle()
tsize=Dataload.nbSamples
trsize=math.floor(Dataload.nbSamples*0.8)
print(trsize)
nclass=Dataload.nbClass
print(nclass)
imsize=32
shownb=256

--trainData = nn.DataList()
--trainData:appendDataSet(data,'00')
--print(trainDataload.nbSample)

trainData = {
   data = torch.Tensor(trsize, 3, imsize, imsize),
   labels = torch.Tensor(trsize, 1),--nclass):fill(-1),
   size = function() return trsize end
}

--[[
   image.display(Dataload[2][1])
   image.display(Dataload[4][1])
   image.display(Dataload[6][1])
   image.display(Dataload[8][1])
   image.display(Dataload[10][1])
]]
--print(#Dataload[1][2])
--sleep(3)

--print(#Dataload[1][1])
for t = 1, trsize do
   trainData.data[t][1] = image.scale(Dataload[t][1][1], imsize, imsize)
   trainData.data[t][2] = image.scale(Dataload[t][1][2], imsize, imsize)
   trainData.data[t][3] = image.scale(Dataload[t][1][3], imsize, imsize)
   for o=1, nclass do
      if Dataload[t][2][o]==1 then
	 trainData.labels[t][1] = o
      end
   end
   --[[
      if Dataload[t][2][1]==1 then
      trainData.labels[t] = 1
      elseif Dataload[t][2][2]==1 then
      trainData.labels[t] = 2
      end
   ]]
   --trainData.labels[t-trsize] =  Dataload[t][2]
   xlua.progress(t, trainData:size())
end
--f256S = trainData.data[{{1,shownb}}]
--image.display{image=f256S, nrow=math.sqrt(shownb), nrow=(shownb), padding=2, zoom=1, legend='Input images'}
--print(trainData.labels[{{1,shownb}}])

tssize=tsize-trsize

-- Finally we load the test data.
testData = {
   data = torch.Tensor(tssize, 3, imsize, imsize),
   labels = torch.Tensor(tssize, 1),--nclass):fill(-1),
   size = function() return tssize end
}

for t = trsize+1, tsize do
   testData.data[t-trsize][1] = image.scale(Dataload[t][1][1], imsize, imsize)
   testData.data[t-trsize][2] = image.scale(Dataload[t][1][2], imsize, imsize)
   testData.data[t-trsize][3] = image.scale(Dataload[t][1][3], imsize, imsize)
   for o=1, nclass do
      if Dataload[t][2][o]==1 then
	 testData.labels[t-trsize][1] = o
      end
   end
   --testData.labels[t-trsize] =  Dataload[t][2]
   xlua.progress(t-trsize, testData:size())
end
--f256S = testData.data[{{1,shownb}}]
--image.display{image=f256S, nrow=math.sqrt(shownb), padding=2, zoom=1, legend= 'Test Images'}
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
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

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
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, normkernel):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
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
--[[
if opt.visualize then
   first256Samples_y = trainData.data[{ {1,256},1 }]
   first256Samples_u = trainData.data[{ {1,256},2 }]
   first256Samples_v = trainData.data[{ {1,256},3 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: U channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: V channel'}
end
]]
print(trainData.labels[{{1, 256},{}}])

--sleep(5)
