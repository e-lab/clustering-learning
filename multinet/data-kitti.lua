------------------------------------------------------------------------------
-- This script loads the KITTI dataset
-- training data, and pre-process it to facilitate learning.
-- Aysegul Dundar & Alfredo Canziani
-- Date : 04/28/2013 & May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'   -- to visualize the dataset
require 'sys'
require 'nnx'
require 'ffmpeg'
require 'xml'
require 'kitti2Dbox'
require 'eex'

-- Exporting functions to the global namespace -------------------------------
local max = math.max
local min = math.min
local random = math.random
local floor = math.floor
local zeros = torch.zeros

-- Title ---------------------------------------------------------------------
print [[
********************************************************************************
>>>>>>>>>>>>>>>>>>>>>>> Loading KITTI (car) dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<
********************************************************************************
]]

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text('Opions')
   cmd:option('-visualize',  true,    'display kernels')
   cmd:option('-seed',        1,      'initial random seed')
   cmd:option('-threads',     8,      'threads')
   cmd:option('-savedataset', true,   'save dataset')
   cmd:option('-width',       46,     'width of extracted patch')
   cmd:option('-height',      46,     'height of extracted patch')
   cmd:option('-ratio',       0.8,    'ratio of train to test dataset split')
   cmd:option('-samplepercar',6,      'number of the patch to extract from per car (bounding box)')
   cmd:option('-maxBg',       false,  'max number of background samples')

   cmd:text()
   opt = cmd:parse(arg or {}) -- pass parameters to training files:
end

opt = opt or {}
opt.seed = opt.seed or 1
opt.threads = opt.threads or 3
opt.height = opt.height or 46
opt.width  = opt.width  or 46
opt.ratio = opt.ratio or .8
opt.samplepercar = opt.samplepercar or 6

-- Parameters ----------------------------------------------------------------
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
local ds = eex.datasetsPath()
local path = ds .. 'KITTI_dataset/city/'

-- Global functions ----------------------------------------------------------
-- Parse XML
function parseXML(tracklet_labels)
   local parse = xml.parse(tracklet_labels)
   local tracklet = parse.boost_serialization.tracklets

   return tracklet
end

-- In car (explanation??? what the hell is this???)
function InCar(x1,x2,y1,y2)
   inCar = false
   for i, detect in ipairs(detections) do
      if  (detect.objectType == 'Car') and not (detect.x1==1 and detect.y1==1 and detect.x2==iwidth and detect.y2==iheight) then
         if (x1>=detect.x1 and x1<=detect.x2) or (x2>=detect.x1 and x2<=detect.x2) or (y1>=detect.y1 and y1<=detect.y2) or (y2>=detect.y1 and y2<=detect.y2) then
            inCar = true
         end
      end
   end
   return inCar
end

-- Extract patches
function extractPatches(dspath, tracklet)
   videoframes = #sys.dir(dspath)-2 -- #sys.dir(dspath) == total number of frames in video dump (minum . and ..)
   for imgi = 1,videoframes do
      rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi-1)..'.png'))
      -- get bounding boxes from tracklets:
      detections = {}

      for k=1, tracklet.count do
         first = tonumber(tracklet.item[k].first_frame)
         count = tonumber(tracklet.item[k].poses.count)+first
         if  first<imgi and imgi<=count then
            w=tracklet.item[k].w
            h=tracklet.item[k].h
            l=tracklet.item[k].l
            box = kitti2Dbox(tracklet.item[k].poses.item[imgi-first])
            box.objectType = tracklet.item[k].objectType

            iwidth = rawFrame:size(3)
            iheight = rawFrame:size(2)

            box.x1 = max(1, min(iwidth, box.x1))
            box.y1 = max(1, min(iheight, box.y1))
            box.x2 = max(1, min(iwidth, box.x2))
            box.y2 = max(1, min(iheight, box.y2))

            if not (box.x1==1 and box.x2==iwidth and box.y1==1 and box.y2==iheight) then
               table.insert(detections, box)
            end
         end
      end
      ncar = 0
      for i, detect in ipairs(detections) do
         for j=1, opt.samplepercar do
            if (detect.objectType == 'Car') then
               if(detect.x1+opt.width*3<=detect.x2) and (detect.y1+opt.height*3<=detect.y2) then
                  x = random(detect.x1+opt.width*3/2,detect.x2-opt.width*3/2)
                  y = random(detect.y1+opt.height*3/2,detect.y2-opt.height*3/2)

                  randompatch = rawFrame[{{},{y-opt.height*3/2,y+opt.height*3/2-1},{x-opt.width*3/2,x+opt.width*3/2-1}}]
                  sizepatch = nn.SpatialReSampling{owidth=opt.width,
                  oheight=opt.height}:forward(randompatch)
                  table.insert(carData.data, sizepatch)
                  table.insert(carData.labels, 1)  -- car

                  ncar = ncar+1
               end
            end
         end
      end


      for i=1,ncar+4 do
         x = random(opt.width*3/2+1, iwidth-opt.width*3/2-1)
         y = random(opt.height*3/2+1, iheight-opt.height*3/2-1)
         i = 0
         while (InCar(x-opt.width*3/2,x+opt.width*3/2,y-opt.height*3/2,y+opt.height*3/2)) do
            x = random(opt.width*3/2+1, iwidth-opt.width*3/2-1)
            y = random(opt.height*3/2+1, iheight-opt.height*3/2-1)
            i = i+1
            if (i==50) then
               break
            end
         end
         if (i~=50) then
            randompatch = rawFrame[{{},{y-opt.height*3/2,y+opt.height*3/2-1},{x-opt.width*3/2,x+opt.width*3/2-1}}]
            sizepatch = nn.SpatialReSampling{owidth=opt.width,
            oheight=opt.height}:forward(randompatch)
            table.insert(backgroundData.data, sizepatch)
            table.insert(backgroundData.labels, 0)  -- bg
         end
      end
   end

end

-- Main program -------------------------------------------------------------
-- define dataset tables
carData = {
   data = {},
   labels = {},
   size = function() return #carData.labels end
}
backgroundData = {
   data = {},
   labels = {},
   size = function() return #backgroundData.labels end
}

print '==> loading KITTI tracklets and parsing the XML files'
iter = 0
tot = 13

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0001/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0001/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0002/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0002/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0005/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0005/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0009/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0009/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0011/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0011/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0013/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0013/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0017/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0017/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0018/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0018/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0048/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0048/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0057/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0057/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0059/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0059/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0084/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0084/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1

xlua.progress(iter,tot)
dspath = path .. '2011_09_26_drive_0091/image_02/data/'--/0000000000.png' -- Right images
tracklet_labels = xml.load(path .. '2011_09_26_drive_0091/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)
iter = iter + 1
xlua.progress(iter,tot)


print '==> split dataset into train/test datasets'

local carTrSize = math.floor(opt.ratio*carData:size())
local carTeSize = math.floor((carData:size()-carTrSize))
local shuffleCar = torch.randperm(carData:size())

local shuffleBg = torch.randperm(backgroundData:size())
if opt.maxBg then
   lower = (opt.maxBg < backgroundData:size()) and opt.maxBg or backgroundData:size()
   backgroundData.size = function() return lower end
end
local bgTrSize = math.floor(opt.ratio*backgroundData:size())
local bgTeSize = math.floor((backgroundData:size()-bgTrSize))

trSize = carTrSize + bgTrSize
teSize = carTeSize + bgTeSize

-- Training dataset
trainData = {
   data   = zeros(trSize, 3, opt.width, opt.width),
   labels = zeros(trSize),
   size   = function() return trSize end
}
for i = 1,bgTrSize do
   trainData.data[i] = backgroundData.data[shuffleBg[i]]
end
for i=bgTrSize+1, trSize  do
   trainData.data[i]   = carData.data[shuffleCar[i-bgTrSize]]
   trainData.labels[i] = 1
end
-- display some examples:
image.display{image=trainData.data[{{bgTrSize+1-64,bgTrSize+64}}], nrow=16, zoom=2, legend = 'Train data'}

-- Testing dataset
testData  = {
   data   = zeros(teSize, 3, opt.width, opt.width),
   labels = zeros(teSize),
   size   = function() return teSize end
}
for i = 1,bgTeSize do
   testData.data[i] = backgroundData.data[shuffleBg[i+bgTrSize]]
end
for i=bgTeSize+1, teSize  do
   testData.data[i]   = carData.data[shuffleCar[i-bgTeSize+carTrSize]]
   testData.labels[i] = 1
end
-- display some examples:
image.display{image=testData.data[{{bgTeSize+1-64,bgTeSize+64}}], nrow=16, zoom=2, legend = 'Test data'}


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
