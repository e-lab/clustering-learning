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

   print '==> creating a new dataset from raw files:'

   -- video dataset to get background from:
   local dspath = '../../datasets/driving1.mov'
   local source = ffmpeg.Video{path=dspath, width = 160, height = 120, encoding='jpg', 
   fps=24, loaddump=false, load=false}

   local rawFrame = source:forward()
   -- input video params:
   --local ivch = rawFrame:size(1) -- channels
   ivhe = rawFrame:size(2) -- height
   ivwi = rawFrame:size(3) -- width

   local ivch = 3 -- color channels in images
   local desImaX = 46 -- desired cropped dataset image size
   local desImaY = 46

   local cropTrX = 45 -- desired offset to crop images from train set
   local cropTrY = 48
   local cropTeX = 33 -- desired offset to crop images from test set
   local cropTeY = 35

   local labelPerson = 1 -- label for person and background:
   local labelBg = 2

   local trainDir = '../../datasets/INRIAPerson/96X160H96/Train/pos/'
   local trainImaNumber = #ls(trainDir)
   local testDir = '../../datasets/INRIAPerson/70X134H96/Test/pos/'
   local testImaNumber = #ls(testDir)

   -- dataset size:
   local dataMultiplier = 1 -- optional: take multiple samples per image: +/- 2 pix H, V = 4 total
   trSize = dataMultiplier * trainImaNumber
   teSize = dataMultiplier * testImaNumber

   trainData = {
      data = torch.Tensor(trSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(trSize),
      size = function() return trSize end
   }

   -- load person data:
   for i = 1, trainImaNumber, 2 do
      img = image.loadPNG(trainDir..ls(trainDir)[i],ivch)
      trainData.data[i] = image.crop(img, cropTrX-desImaX/2, cropTrY-desImaY/2, 
      cropTrX+desImaX/2, cropTrY+desImaY/2):clone()
      trainData.labels[i] = labelPerson

      -- load background data:
      img = source:forward()
      local x = math.random(1, ivwi-desImaX+1)
      local y = math.random(15, ivhe-desImaY+1-30) -- added # to get samples more or less from horizon
      trainData.data[i+1] = img[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
      trainData.labels[i+1] = labelBg
   end
   -- display some examples:
   image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}


   testData = {
      data = torch.Tensor(teSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(teSize),
      size = function() return teSize end
   }

   -- load person data:
   for i = 1, testImaNumber, 2 do
      img = image.loadPNG(testDir..ls(testDir)[i],ivch)
      testData.data[i] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2, 
      cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
      testData.labels[i] = labelPerson

      -- load background data:
      img = source:forward()
      local x = math.random(1,ivwi-desImaX+1)
      local y = math.random(15,ivhe-desImaY+1-30) -- added # to get samples more or less from horizon
      testData.data[i+1] = img[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
      testData.labels[i+1] = labelBg
   end
   -- display some examples:
   image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

   --save created dataset:
   torch.save('../../datasets/INRIAPerson/train.t7',trainData)
   torch.save('../../datasets/INRIAPerson/test.t7',testData)
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

trainData.size = function() return trSize end
testData.size = function() return teSize end

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
