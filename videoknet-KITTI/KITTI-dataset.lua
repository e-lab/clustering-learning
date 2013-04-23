----------------------------------------------------------------------
-- KITTI dataset interface
--
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
torch.setdefaulttensortype('torch.DoubleTensor')
--if not qt then
--   opt.visualize = false
--end

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
tracklet_labels = xml.load('../../datasets/KITTI/2011_09_26_drive_0060/tracklet_labels.xml')
tracklet = parseXML(tracklet_labels)



----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

dspath = '../../datasets/KITTI/2011_09_26_drive_0060/image_02/data/'--/0000000000.png' -- Right images
imgi = 0
rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi)..'.png'))
print(rawFrame:size())



win = image.display{image=rawFrame, zoom=1}
videoframes = #sys.dir(dspath)-2 -- #sys.dir(dspath) == total number of frames in video dump (minum . and ..)
for imgi = 1,videoframes do
	rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi-1)..'.png'))
	image.display{image=rawFrame, win=win, zoom=0.5}

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
     if (detect.objectType == 'Car') then  win.painter:setcolor('green')
     elseif (detect.objectType == 'Cyclist') then  win.painter:setcolor('blue')
     else win.painter:setcolor('red') 
     end
 
     win.painter:rectangle(math.floor(detect.x1-1), detect.y1-1, detect.x2-detect.x1+1, detect.y2-detect.y1+1)
     win.painter:stroke()
     win.painter:setfont(qt.QFont{serif=false,italic=false,size=16})
<<<<<<< HEAD
     win.painter:moveto(detect.x1, detect.y1)
     win.painter:show(detect.objectType)

  end
=======
     win.painter:moveto(detections.x1, detections.y1)
     win.painter:show(tracklet.item[1].objectType)
     
     sys.sleep(0.5)
>>>>>>> sleep function to slow down demo

end


-- normalize and prepare dataset:
--[[neighborhood = image.gaussian1D(9)
normalization = nn.SpatialContrastiveNormalization(ivch, neighborhood, 1e-3)

function createDataBatch()
   trainData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
   for i = 1, nfpr do -- just get a few frames to begin with
      procFrame = normalization:forward(rawFrame) -- full LCN!
      trainData[i] = procFrame
      rawFrame = source:forward()
      -- do a live display
      winm = image.display{image=procFrame, win=winm}
   end
   return trainData
end

createDataBatch()]]
