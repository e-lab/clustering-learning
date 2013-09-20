------------------------------------------------------------------------------
-- Loading stop signs
------------------------------------------------------------------------------
-- Alfredo Canziani Sep 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'xlua'

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>>>>>> Loading SIFTFLOW dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Parameters ----------------------------------------------------------------
opt = opt or {}
local height = opt.height or 46
local width = opt.width or 46
torch.setdefaulttensortype('torch.FloatTensor')

-- Main program -------------------------------------------------------------
print '==> Building (or loading from cache) SIFTFLOW dataset'
require 'load-siftflow'

trSize = trainData:size()
teSize = testData:size()

print '==> Allocating RAM for SIFTFLOW training and testing datasets'
newTrainData = {
   data = torch.Tensor(trSize,3,height,width),
   labels = torch.zeros(trSize),
   size = function() return trSize end
}

newTestData = {
   data = torch.Tensor(teSize,3,height,width),
   labels = torch.zeros(teSize),
   size = function() return teSize end
}

print '==> Loadin training dataset to RAM'
for idx = 1, trSize do
   local a = trainData[idx]
   newTrainData.data[idx] = a[1][{ {},{a[4]-23+1,a[4]+23},{a[3]-23+1,a[3]+23} }]
   newTrainData.labels[idx] = a[2] * l - 1
   xlua.progress(idx,trSize)
end

trainData = newTrainData
newTrainData = nil

-- display some examples:
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}

print '==> Loading testing dataset to RAM'
for idx = 1, teSize do
   local a = testData[idx]
   newTestData.data[idx] = a[1][{ {},{a[4]-23+1,a[4]+23},{a[3]-23+1,a[3]+23} }]
   newTestData.labels[idx] = a[2] * l - 1
   xlua.progress(idx,teSize)
end

testData  = newTestData
newTestData = nil

-- display some examples:
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

-- Preprocessing -------------------------------------------------------------
if opt.preprocessing=='norm(y)+norm(u)+norm(v)' then dofile 'preprocessing.lua' end

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
