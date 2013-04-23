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
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to training files:


torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

function parseXML(tracklet_labels)
  print '==> parsing the XML file'
  parse = xml.parse(tracklet_labels)
  tracklet = parse.boost_serialization.tracklets

  return tracklet  
end
----------------------------------------------------------------------
print '==> test KITTI dataset'

----------------------------------------------------------------------
print '==> load KITTI tracklets'
tracklet_labels = xml.load('../../datasets/KITTI/2011_09_26_drive_0005/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)



----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

dspath = '../../datasets/KITTI/2011_09_26_drive_0005/image_02/data/'--/0000000000.png' -- Right images
imgi = 0
rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi)..'.png'))


-- define dataset tables
trainData = {
   data = {},
   labels = {},
   size = function() return #trainData.labels end
}
testData = {
   data = {},
   labels = {},
   size = function() return #testData.labels end
}

w_patch = 32
h_patch = 32
d_patch = 3

-- shorcuts
local max = math.max
local min = math.min
local random = math.random


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
        table.insert(detections, box)-- print(detections) 
	   end
    end

    for i, detect in ipairs(detections) do
      if (detect.objectType == 'Car') then 

        iwidth = rawFrame:size(3)
        iheight = rawFrame:size(2)
        detect.x1 = max(1, min(iwidth, detect.x1))
        detect.y1 = max(1, min(iheight, detect.y1))
        detect.x2 = max(1, min(iwidth, detect.x2))
        detect.y2 = max(1, min(iheight, detect.y2))

        if (detect.x1+w_patch<=detect.x2) and (detect.y1+h_patch<=detect.y2) then
           x = math.random(detect.x1+w_patch/2,detect.x2-w_patch/2)
           y = math.random(detect.y1+h_patch/2,detect.y2-h_patch/2)

           randompatch = rawFrame[{{},{y-h_patch/2,y+h_patch/2},{x-w_patch/2,x+w_patch/2}}]
           table.insert(trainData.data, randompatch)
           table.insert(trainData.labels, 1)  -- car

        end 
      end
    end
end

   image.display{image=trainData.data, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}
