------------------------------------------------------------------------------
-- Loading stop signs
------------------------------------------------------------------------------
-- Alfredo Canziani Jun 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'
require 'sys'
require 'pl'
require 'eex'
require 'xlua'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>>>>>> Loading GTSRB stop signs <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Parameters ----------------------------------------------------------------
local ds = eex.datasetsPath()
local path = ds .. 'GTSRB/'
local trPath = path .. 'Final_Training/Images/'
local tePath = path .. 'Final_Test/Images/'
opt = opt or {}
local height = opt.height or 46
local width = opt.width or 46
local trNames = ls(trPath .. '00014/*png')
trSize = 2 * #trNames
local translate = 5

-- Main program -------------------------------------------------------------
print '==> creating a new training dataset from raw files:'
trainDataTemp = {
   data = torch.Tensor(trSize,3,height+2*translate,width+2*translate),
   labels = torch.zeros(trSize),
   size = function() return trSize end
}

for i = 1,trSize/2 do
   local img = image.load(trNames[i])
   local w,h = (#img)[3],(#img)[2]
   local min = (w < h) and w or h
   for j = -1,0 do
      -- x = math.random(0,w-min)
      -- y = math.random(0,h-min)
      x = (j<0) and 0 or w-min
      y = (j<0) and 0 or h-min
      local imgCrp = image.crop(img,x,y,x+min,y+min)
      image.scale(trainDataTemp.data[i*2+j],imgCrp)
   end
   xlua.progress(i*2,trSize)
end

print '       Translation/multiplication of the dataset'
trainData = {
   data = torch.Tensor(trSize*9,3,height,width),
   labels = torch.zeros(trSize*9),
   size = function() return trSize*9 end
}

local idx = 0
for i = 1,trSize do
   for xCrop = 0,translate*2,translate do
      for yCrop = 0,translate*2,translate do
         idx = idx + 1
         trainData.data[idx] = image.crop(trainDataTemp.data[i],xCrop,yCrop,xCrop+width,yCrop+height)
         xlua.progress(idx,trSize*9)
      end
   end
end

trSize = trainData.size()

-- display some examples:
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}

print '\n==> creating a new testing dataset from raw files:'

local fileName = ls(tePath .. '../*.csv')
local testDataFile = io.open(fileName[1], 'rb')
local line = testDataFile:read() -- skipping the header

teSize = 0
while line ~= nil do
   if tonumber(sys.split(line,';')[8]) == 14 then teSize = teSize + 1 end
   line = testDataFile:read()
end
teSize = teSize*2

local testDataTemp = {
   data = torch.Tensor(teSize,3,height+2*translate,width+2*translate),
   labels = torch.zeros(teSize),
   size = function() return teSize end
}

testDataFile:seek('set',0)
line = testDataFile:read()
local count = 0
while line ~= nil do
   if tonumber(sys.split(line,';')[8]) == 14 then
      local img = image.load(tePath .. sys.split(line,';')[1])
      local w,h = (#img)[3],(#img)[2]
      local min = (w < h) and w or h
      for j = -1,0 do
         -- x = math.random(0,w-min)
         -- y = math.random(0,h-min)
         x = (j<0) and 0 or w-min
         y = (j<0) and 0 or h-min
         local imgCrp = image.crop(img,x,y,x+min,y+min)
         count = count + 1
         image.scale(testDataTemp.data[count],imgCrp)
      end
      xlua.progress(count,teSize)
   end
   line = testDataFile:read()
end

print '       Translation/multiplication of the dataset'
testData = {
   data = torch.Tensor(teSize*9,3,height,width),
   labels = torch.zeros(teSize*9),
   size = function() return teSize*9 end
}

local idx = 0
for i = 1,teSize do
   for xCrop = 0,translate*2,translate do
      for yCrop = 0,translate*2,translate do
         idx = idx + 1
         testData.data[idx] = image.crop(testDataTemp.data[i],xCrop,yCrop,xCrop+width,yCrop+height)
         xlua.progress(idx,teSize*9)
      end
   end
end

teSize = testData.size()

-- display some examples:
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
