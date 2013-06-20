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
--lim                (default 50)      At least <lim> examples per sign, max 1000
]]
end

-- Parameters ----------------------------------------------------------------
local ds = eex.datasetsPath()
local path = ds .. 'LISA/'
opt = opt or {}
local height = opt.height or 46
local width = opt.width or 46
opt.ratio = opt.ratio or .8
local lim = opt.lim or 50

-- Main program -------------------------------------------------------------
print '==> Loading labels and choosing appropriate signs'

local annotationsFile = io.open(path .. 'allAnnotations.csv', 'rb')
local header = sys.split(annotationsFile:read(),';')
local line
for i = 1, 4623 do line = annotationsFile:read() end -- skipping the B&W images
local readFrom = annotationsFile:seek()
line = annotationsFile:read()
local humanLabels = {}
local idx = 0
while line ~= nil do
   annotation = sys.split(line,';')
   idx = idx + 1
   humanLabels[idx] = annotation[2]
   line = annotationsFile:read()
   xlua.progress(idx,3232)
end

local signList = {}
for i,s in ipairs(humanLabels) do
   if signList[s] == nil then signList[s] = 1 else signList[s] = signList[s] + 1 end
end
-- torch.save('signList.t7',signList,'ascii')
local datasetSize = 0
for k in pairs(signList) do
   if signList[k] >= lim then datasetSize = datasetSize + signList[k] end
end

local signArray = {}
for s in pairs(signList) do signArray[#signArray + 1] = s end
table.sort(signArray,function(a,b) return signList[a] > signList[b] end)

local revSignDic = {}
for i,s in ipairs(signArray) do revSignDic[s] = i end

print('==> Loading file names and cropping coordinates (at least ' .. lim .. ' samples)')

local dataset = {
   data = torch.Tensor(datasetSize,3,height,width),
   labels = torch.Tensor(datasetSize),
   size = function() return datasetSize end
}

annotationsFile:seek('set',readFrom)
line = annotationsFile:read()
idx = 0
humanLabels = {}
while line ~= nil do
   annotation = sys.split(line,';')
   if signList[annotation[2]] >= lim then
      local w,h = (annotation[5] - annotation[3]), (annotation[6] - annotation[4])
      local min = (w < h) and w or h
      img = image.crop(image.load(path .. annotation[1]),
      annotation[3] + math.floor((w-min)/2), annotation[4] + math.floor((h-min)/2),
      annotation[5] - math.ceil ((w-min)/2), annotation[6] - math.ceil ((h-min)/2))
      idx = idx + 1
      humanLabels[idx] = annotation[2]
      image.scale(img,dataset.data[idx])
   end
   line = annotationsFile:read()
   xlua.progress(idx,dataset.size())
end

print '==> Generating dataset numerical labels'

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

   print '   signList ='
   print(signList)

end

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData
}
