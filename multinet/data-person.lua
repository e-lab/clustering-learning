----------------------------------------------------------------------
-- This script loads the INRIA person dataset
-- training data, and pre-process it to facilitate learning.
-- E. Culurciello & Alfredo Canziani
-- April 2013 & May 2013
----------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'   -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require 'unsup'
require 'sys'
require 'ffmpeg'
require 'eex'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>> Loading INRIA (person) dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('INRIA Person Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- Parameters ----------------------------------------------------------------
local ivch = 3 -- color channels in images
local desImaX = 46 -- desired cropped dataset image size
local desImaY = 46

local cropTrX = 45 -- desired offset to crop images from train set
local cropTrY = 48
local cropTeX = 33 -- desired offset to crop images from test set
local cropTeY = 35

local labelPerson = 0 -- label for person and background:

local ds = eex.datasetsPath()
local path = ds .. 'INRIAPerson/'
local trainDir = path .. '96X160H96/Train/pos/'
local trainImaNumber = #ls(trainDir)
local testDir = path .. '70X134H96/Test/pos/'
local testImaNumber = #ls(testDir)

-- dataset size:
local dataMultiplier = 6 -- optional: take multiple samples per image: +/|/- 5px H, |/+ 5px V, mirror = 3x2x2 = 12 total
trSize = dataMultiplier * trainImaNumber / 1 -- 1 == mirror, 2 == no mirror
teSize = dataMultiplier * testImaNumber / 1

-- Main program -------------------------------------------------------------
print '==> creating a new training dataset from raw files:'

trainData = {
   data = torch.Tensor(trSize, ivch,desImaX,desImaY),
   labels = torch.Tensor(trSize),
   size = function() return trSize end
}

-- load person data:
local idx = 0
for i = 1, trainImaNumber, 1 do
   local img = image.loadPNG(trainDir..ls(trainDir)[i],ivch)
   for cropTrX = cropTrX-5, cropTrX+5, 5 do
      for cropTrY = cropTrY, cropTrY+5, 5 do
         idx = idx + 1
         -- trainData.data[idx] = image.scale(image.crop(img, cropTrX-(desImaX-10)/2, cropTrY-(desImaY-10)/2, cropTrX+(desImaX-10)/2, cropTrY+(desImaY-10)/2),desImaX,desImaY)
         trainData.data[idx] = image.crop(img, cropTrX-desImaX/2, cropTrY-desImaY/2, cropTrX+desImaX/2, cropTrY+desImaY/2)
         trainData.labels[idx] = labelPerson
         xlua.progress(idx,trSize)
      end
   end
end
-- display some examples:
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}

print '==> creating a new testing dataset from raw files:'

testData = {
   data = torch.Tensor(teSize, ivch,desImaX,desImaY),
   labels = torch.Tensor(teSize),
   size = function() return teSize end
}

-- load person data:
idx = 0
for i = 1, testImaNumber, 1 do
   local img = image.loadPNG(testDir..ls(testDir)[i],ivch)
   for cropTeX = cropTeX-5, cropTeX+5, 5 do
      for cropTeY = cropTeY, cropTeY+5, 5 do
         idx = idx + 1
         -- testData.data[idx] = image.scale(image.crop(img, cropTeX-(desImaX-10)/2, cropTeY-(desImaY-10)/2, cropTeX+(desImaX-10)/2, cropTeY+(desImaY-10)/2),desImaX,desImaY)
         testData.data[idx] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2, cropTeX+desImaX/2, cropTeY+desImaY/2)
         testData.labels[idx] = labelPerson
         xlua.progress(idx,teSize)
      end
   end
end
-- display some examples:
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

-- Shuffling the datasets and uniforming the statistics ---------------------
print '==> Uniforming statistics: #training = 14040, #testing = 4860 (such as for <stop>)'
local shuffle = torch.randperm(trainData:size())
oldTrainData = trainData
trSize = 14040
trainData = {
   data = torch.Tensor(trSize, ivch,desImaX,desImaY),
   labels = torch.Tensor(trSize),
   size = function() return trSize end
}
for i = 1,trSize do
   trainData.data[i] = oldTrainData.data[shuffle[i]]
   trainData.labels[i] = oldTrainData.labels[shuffle[i]]
end
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'New Train Data'}

local shuffle = torch.randperm(testData:size())
oldTestData = testData
teSize = 4860
testData = {
   data = torch.Tensor(teSize, ivch,desImaX,desImaY),
   labels = torch.Tensor(teSize),
   size = function() return teSize end
}
for i = 1,teSize do
   testData.data[i] = oldTestData.data[shuffle[i]]
   testData.labels[i] = oldTestData.labels[shuffle[i]]
end
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'New Test Data'}

-- Displaying the dataset architecture ---------------------------------------
print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
