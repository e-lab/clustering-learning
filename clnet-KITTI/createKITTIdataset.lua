----------------------------------------------------------------------
-- KITTI dataset interface
-- 
--
-- Author : Aysegul Dundar 
-- email : adundar@purdue.edu
-- Date 04/23/2013
----------------------------------------------------------------------

require 'nnx'
require 'image'
require 'ffmpeg'
require 'xml' 
require 'kitti2Dbox'

cmd = torch.CmdLine()
cmd:text('Opions')
cmd:option('-visualize',  true,    'display kernels')
cmd:option('-seed',        1,      'initial random seed')
cmd:option('-threads',     8,      'threads')
cmd:option('-savedataset', true,  'save dataset')
cmd:option('-width',       46*3,   'width of extracted patch')
cmd:option('-height',      46*3,   'height of extracted patch')
cmd:option('-down',        3,      'downsample of the extracted patch')
cmd:option('-ratio',       0.8,    'ratio of train to test dataset split')
cmd:option('-samplepercar',6,      'number of the patch to extract from per car (bounding box)')

cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to training files:


torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

-- shorcuts
local max = math.max
local min = math.min
local random = math.random
local floor = math.floor
local zeros = torch.zeros



function parseXML(tracklet_labels)
  print '==> parsing the XML file'
  parse = xml.parse(tracklet_labels)
  tracklet = parse.boost_serialization.tracklets

  return tracklet  
end


function InCar(x1,x2,y1,y2)
   inCar = false
   for i, detect in ipairs(detections) do
      if  (detect.objectType == 'Car') and not (detect.x1==1 and detect.y1==1 and detect.x2==iwidth and detect.y2==iheight) then 
         if (x1>=detect.x1 and x1<=detect.x2) or (x2>=detect.x1 and x2<=detect.x2) or (y1>=detect.y1 and y1<=detect.y2) or (y2>=detect.y1 and y2<=detect.y2) then
           inCar = true  end
      end  
   end    
   return inCar
end


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
             if(detect.x1+opt.width<=detect.x2) and (detect.y1+opt.height<=detect.y2) then
               x = random(detect.x1+opt.width/2,detect.x2-opt.width/2)
               y = random(detect.y1+opt.height/2,detect.y2-opt.height/2)
          
               randompatch = rawFrame[{{},{y-opt.height/2,y+opt.height/2-1},{x-opt.width/2,x+opt.width/2-1}}]
               sizepatch = nn.SpatialReSampling{owidth=opt.width/opt.down,
                                      oheight=opt.height/opt.down}:forward(randompatch)
               table.insert(carData.data, sizepatch)
               table.insert(carData.labels, 1)  -- car
      
               ncar = ncar+1
             end 
          end
        end
     end


     for i=1,ncar+4 do
        x = random(opt.width/2+1, iwidth-opt.width/2-1)
        y = random(opt.height/2+1, iheight-opt.height/2-1)
        i = 0
        while (InCar(x-opt.width/2,x+opt.width/2,y-opt.height/2,y+opt.height/2)) do
           x = random(opt.width/2+1, iwidth-opt.width/2-1)
           y = random(opt.height/2+1, iheight-opt.height/2-1)
           i = i+1
           if (i==50) then 
              break 
           end
        end
        if (i~=50) then
           randompatch = rawFrame[{{},{y-opt.height/2,y+opt.height/2-1},{x-opt.width/2,x+opt.width/2-1}}]
           sizepatch = nn.SpatialReSampling{owidth=opt.width/opt.down,
                                      oheight=opt.height/opt.down}:forward(randompatch)
           table.insert(backgroundData.data, sizepatch)
           table.insert(backgroundData.labels, 0)  -- bg
        end
     end
  end

end


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


----------------------------------------------------------------------


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0001/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0001/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0002/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0002/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0005/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0005/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0009/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0009/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0011/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0011/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0013/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0013/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0017/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0017/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0018/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0018/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0048/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0048/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)



dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0057/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0057/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0059/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0059/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0084/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0084/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

dspath = '../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0091/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../elabshare/datasets/KITTI_dataset/city/2011_09_26_drive_0091/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)

--[[dspath = '../../datasets/KITTI/2011_09_26_drive_0060/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../datasets/KITTI/2011_09_26_drive_0060/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


dspath = '../../datasets/KITTI/2011_09_26_drive_0014/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../datasets/KITTI/2011_09_26_drive_0014/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)   14 give problem about tracklet a nil value 


dspath = '../../datasets/KITTI/2011_09_26_drive_0056/image_02/data/'--/0000000000.png' -- Right images
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../datasets/KITTI/2011_09_26_drive_0056/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)
extractPatches(dspath, tracklet)


]]
-- 60 is giving problems with the opening image 78

print('car', carData:size())
print('bg', backgroundData:size())

--[[local f16S1 = torch.Tensor(2100, 3, opt.width/opt.down, opt.width/opt.down)
local f16S2 = torch.Tensor(2100, 3, opt.width/opt.down, opt.width/opt.down)
for i=1, 2100 do 
--  k = random(1, carData:size())
  f16S1[i]=carData.data[i]  
  k = random(1, backgroundData:size())
  f16S2[i]=backgroundData.data[i]  
end


image.display{image=f16S1, nrow=70, ncol=30, padding=2, zoom=1, 
            legend='Patches for car dataset'}
image.display{image=f16S2, nrow=70, ncol=30, padding=2, zoom=1, 
            legend='Patches for background dataset'}


collectgarbage()
]]

print '==> split dataset into train/test datasets'


local nTrainData  = floor(opt.ratio*carData:size()/2)*2  -- we want it to be dividable by 2
local nTestData = floor((carData:size()-nTrainData)/2)*2

-- define tables for training/testing dataset
trainData = {
   data   = zeros(nTrainData, 3, opt.width/opt.down, opt.width/opt.down),
   labels = zeros(nTrainData),
   size   = function() return trainData.labels:size() end
}
testData  = {
   data   = zeros(nTestData,3, opt.width/opt.down, opt.width/opt.down),
   labels = zeros(nTestData),
   size   = function() return testData.labels:size() end
}

local shuffleCar = torch.randperm(carData:size())
local shufflebg = torch.randperm(backgroundData:size())


-- split data to train/test sets with ratio of (1-ratio : ratio)
for i=1 , nTrainData/2  do
   trainData.data[2*i-1]   = carData.data[shuffleCar[i]]
   trainData.labels[2*i-1] = 1
   trainData.data[2*i]   = backgroundData.data[shufflebg[i]]
   trainData.labels[2*i] = 2
end 


for i=1 , nTestData/2  do
   testData.data[2*i-1]   = carData.data[shuffleCar[i+nTrainData/2]]
   testData.labels[2*i-1] = 1
   testData.data[2*i]   = backgroundData.data[shufflebg[i+nTrainData/2]]
   testData.labels[2*i] = 2
end 

print '==> display and save'

if opt.savedataset then
   torch.save('trainData-kitti.t7', trainData)
   torch.save('testData-kitti.t7', testData)
end
image.display{image=trainData.data[{{1,120}}], nrow=15, ncol=15}
collectgarbage()

