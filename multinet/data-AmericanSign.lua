------------------------------------------------------------------------------
-- Loading American road signs
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'
require 'pl'
require 'eex'
require 'xlua'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>>>>> Loading LISA (sign) dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> Processing options'
   opt = lapp [[
--ratio              (default 0.8)     Ratio of train to test dataset split
--height             (default 46)      Height of dataset images
--width              (default 46)      Width of dataset images
--visualize                            Show some samples
--stat                                 Show datasets statistics
]]
end

-- Parameters ----------------------------------------------------------------
local ds = eex.datasetsPath()
local path = ds .. 'LISA/'
opt = opt or {}
local height = opt.height or 46
local width = opt.width or 46
opt.ratio = opt.ratio or .8

-- Main program -------------------------------------------------------------
print '==> Loading file names and cropping coordinates'

dataset = {
   data = torch.Tensor(3232,3,height,width),
   labels = torch.Tensor(3232),
   size = function() return 3232 end
}

local annotationsFile = io.open(path .. 'allAnnotations.csv', 'rb')
header = sys.split(annotationsFile:read(),';')
for i = 1, 4624 do line = annotationsFile:read() end -- skipping the B&W images
humanLabels = {}
idx = 0
while line ~= nil do
   annotation = sys.split(line,';')
   local w,h = (annotation[5] - annotation[3]), (annotation[6] - annotation[4])
   local min = (w < h) and w or h
   img = image.crop(image.load(path .. annotation[1]),
                    annotation[3] + math.floor((w-min)/2), annotation[4] + math.floor((h-min)/2),
                    annotation[5] - math.ceil ((w-min)/2), annotation[6] - math.ceil ((h-min)/2))
   idx = idx + 1
   image.scale(img,dataset.data[idx])
   humanLabels[idx] = annotation[2]
   line = annotationsFile:read()
   xlua.progress(idx,dataset.size())
end

print '==> Generating dataset numerical labels'
local signList = {}
for i,s in ipairs(humanLabels) do
   if signList[s] == nil then signList[s] = true end
end

local signArray = {}
for s in pairs(signList) do signArray[#signArray + 1] = s end

local revSignDic = {}
for i,s in ipairs(signArray) do revSignDic[s] = i end

for i = 1,dataset.size() do
   dataset.labels[i] = revSignDic[humanLabels[i]] - 1
end

print '==> split dataset into train/test datasets'
trSize  = math.floor(opt.ratio*dataset:size())
teSize = dataset:size() - trSize
local shuffle = torch.randperm(dataset:size())

-- Training dataset
trainData = {
   data   = torch.zeros(trSize, 3, height, width),
   labels = torch.zeros(trSize),
   size   = function() return trSize  end
}
for i=1 , trSize  do
   trainData.data[i]  = dataset.data[shuffle[i]]
   trainData.labels[i] = dataset.labels[shuffle[i]]
end
-- display some examples:
image.display{image=trainData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Train Data'}

-- Testing dataset
testData  = {
   data   = torch.zeros(teSize, 3, height, width),
   labels = torch.zeros(teSize),
   size   = function() return teSize end
}
for i=1 , teSize  do
   testData.data[i] = dataset.data[shuffle[i+trSize]]
   testData.labels[i] = dataset.labels[shuffle[i+trSize]]
end
-- display some examples:
image.display{image=testData.data[{{1,128}}], nrow=16, zoom=2, legend = 'Test Data'}

-- Displaying the dataset architecture ---------------------------------------
print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'

-- Printing the category statistics ------------------------------------------
if opt.stat then
   print '==> showing class statistics'
   gnuplot.figure(1)
   gnuplot.hist(trainData.labels,43)
   gnuplot.title('Training dataset category statistics')
   gnuplot.figure(2)
   gnuplot.hist(testData.labels,43)
   gnuplot.title('Testing dataset category statistics')
end

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
