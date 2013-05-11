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
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

----------------------------------------------------------------------
-- load or generate new dataset:

if paths.filep('../../datasets/INRIAPerson/train.t7') 
   and paths.filep('../../datasets/INRIAPerson/test.t7') then

   print '==> loading previously generated dataset:'
   trainData = torch.load('../../datasets/INRIAPerson/train.t7')
   testData = torch.load('../../datasets/INRIAPerson/test.t7')

   trSize = trainData.data:size(1)
   teSize = testData.data:size(1)

else

   local ivch = 3 -- color channels in images
   local desImaX = 46 -- desired cropped dataset image size
   local desImaY = 46

   local cropTrX = 45 -- desired offset to crop images from train set
   local cropTrY = 48
   local cropTeX = 33 -- desired offset to crop images from test set
   local cropTeY = 35

   local labelPerson = 1 -- label for person and background:

   local trainDir = '../../datasets/INRIAPerson/96X160H96/Train/pos/'
   local trainImaNumber = #ls(trainDir)
   local testDir = '../../datasets/INRIAPerson/70X134H96/Test/pos/'
   local testImaNumber = #ls(testDir)

   -- dataset size:
   local dataMultiplier = 1 -- optional: take multiple samples per image: +/- 2 pix H, V = 4 total
   trSize = dataMultiplier * trainImaNumber / 2
   teSize = dataMultiplier * testImaNumber / 2

   print '==> creating a new training dataset from raw files:'

   trainData = {
      data = torch.Tensor(trSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(trSize),
      size = function() return trSize end
   }

   -- load person data:
   local idx = 0
   for i = 1, trainImaNumber, 2 do
      idx = idx + 1
      img = image.loadPNG(trainDir..ls(trainDir)[i],ivch)
      trainData.data[idx] = image.crop(img, cropTrX-desImaX/2, cropTrY-desImaY/2, cropTrX+desImaX/2, cropTrY+desImaY/2):clone()
      trainData.labels[idx] = labelPerson
      xlua.progress(idx,trSize)
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
   for i = 1, testImaNumber, 2 do
      idx = idx + 1
      img = image.loadPNG(testDir..ls(testDir)[i],ivch)
      testData.data[idx] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2, cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
      testData.labels[idx] = labelPerson
      xlua.progress(idx,teSize)
   end
   -- display some examples:
   image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

end

-- Displaying the dataset architecture ---------------------------------------
print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'

--[[ Save created dataset ------------------------------------------------------
torch.save('../../datasets/INRIAPerson/train.t7',trainData)
torch.save('../../datasets/INRIAPerson/test.t7',testData)]]

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
