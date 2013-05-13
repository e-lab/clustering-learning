------------------------------------------------------------------------------
-- Loading street signs
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'
require 'sys'
require 'pl'
require 'eex'
require 'xlua'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> Processing options'
   opt = lapp [[
       --playDataset                        visualise the whole signs dataset
       --firstFrame         (default 21)    choose 1st valid frame to load [1, 30]
       --lastFrame          (default 30)    choose last valid frame to load [1, 30]
   -h, --height             (default 46)    height of dataset images
   -w, --width              (default 46)    width of dataset images
       --maxNbPhysicalSigns (default 75)    max number of physical signs to pick from each category
       --checkTestDataset                   check correctness of testing dataset
       --teSize             (default 12630) enter the testing dataset size [1,12630]
]]
   opt.visualize = true
end

-- Parameters ----------------------------------------------------------------
ds = eex.datasetsPath()
path = ds .. 'GTSRB/'
trPath = path .. 'Final_Training/Images/'
tePath = path .. 'Final_Test/Images/'
height = opt.height
width = opt.width
teSize = opt.teSize

-- Main program -------------------------------------------------------------
print '==> Loading human readable labels'
humanReadableDataFile = io.open(path .. 'Categories-name.txt', 'rb')
for i = 1,3 do line = humanReadableDataFile:read() end -- skipping the header
humanLabels = {}
while line ~= nil do
   labels = sys.split(line,'- ')
   table.insert(humanLabels,labels[2])
   line = humanReadableDataFile:read()
end

print '==> creating a new training dataset from raw files:'
totNbSign = 0
nbSign = {}
for i = 1, #ls(trPath) do
   nbSign[i] = #ls(trPath..ls(trPath)[i]..'/*.png')/30
   nbSign[i] = (nbSign[i] < opt.maxNbPhysicalSigns) and nbSign[i] or opt.maxNbPhysicalSigns
   totNbSign = totNbSign + nbSign[i]
end
trSize = totNbSign * (opt.lastFrame - opt.firstFrame + 1)

trainData = {
   data = torch.Tensor(trSize,3,height,width),
   labels = torch.Tensor(trSize),
   size = function() return trSize end
}

-- Load, crop and resize image
idx = 0
for i = 1, #ls(trPath) do -- loop over different signs type
   for j = 1, nbSign[i] do -- loop over different sample of same sign type
      for k = opt.firstFrame, opt.lastFrame do -- loop over different frames of the same physical sign
         img = image.load(string.format('%s%s/%05d_%05d.png',trPath,ls(trPath)[i],j-1,k-1))
         w,h = (#img)[3],(#img)[2]
         min = (w < h) and w or h
         idx = idx + 1
         img  = image.crop(img,math.floor((w-min)/2),math.floor((h-min)/2),w-math.ceil((w-min)/2),h-math.ceil((h-min)/2))
         image.scale(img,trainData.data[idx])
         trainData.labels[idx] = i-1
         xlua.progress(idx,trSize)
      end
   end
end
-- display some examples:
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}

print '==> creating a new testing dataset from raw files:'

testData = {
   data = torch.Tensor(teSize,3,height,width),
   labels = torch.Tensor(teSize),
   size = function() return teSize end
}

fileName = ls(tePath .. '../*.csv')
testDataFile = io.open(fileName[1], 'rb')
line = testDataFile:read() -- skipping the header

for i = 1, teSize do
   img = image.load(string.format('%s%05d.png',tePath,i-1))
   w,h = (#img)[3],(#img)[2]
   min = (w < h) and w or h
   img  = image.crop(img,math.floor((w-min)/2),math.floor((h-min)/2),w-math.ceil((w-min)/2),h-math.ceil((h-min)/2))
   image.scale(img,testData.data[i])
   testData.labels[i] = sys.split(testDataFile:read(),';')[8]
   xlua.progress(i,teSize)
end
-- display some examples:
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

-- Verifying testing dataset
if opt.checkTestDataset then
   for i = 1, teSize do
      win = image.display{image=testData.data[i],zoom=10,win=win}
      io.write(humanLabels[testData.labels[i]+1])
      io.read()
   end
end

-- Play the whole coarse dataset, if requested
if opt.playDataset then
   print 'Visualising the dataset'
   for sign = 1, #ls(trPath) do
      for i = 1,#ls(trPath .. ls(trPath)[sign] .. '/*.png'),1 do
         img = image.load(ls(trPath .. ls(trPath)[sign] .. '/*.png')[i])
         win = image.display{image=img,zoom=10,win=win}
         --io.read()
      end
   end
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

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
