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

  print(tracklet)
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

dspath = '../../datasets/KITTI/2011_09_26_drive_0060/image_03/data/'--/0000000000.png' -- Right images
imgi = 0
rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi)..'.png'))


win = image.display(rawFrame)
videoframes = #sys.dir(dspath)-2 -- #sys.dir(dspath) == total number of frames in video dump (minum . and ..)
for imgi = 1,videoframes do
	rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi-1)..'.png'))
	image.display{image=rawFrame, win=win}

	-- get bounding boxes from tracklets:
	detections = {}
    w=tracklet.item[1].w
    h=tracklet.item[1].h
    l=tracklet.item[1].l

	for i=1, tracklet.item[1].poses.count do
      detections = kitti2Dbox(tracklet.item[1].poses.item[1])
       
	end
 
     win.painter:setcolor(1,0,0)
     win.painter:rectangle(detections.x1, detections.y1, detections.x2-detections.x1, detections.y2-detections.y1)
     win.painter:stroke()
     win.painter:setfont(qt.QFont{serif=false,italic=false,size=16})
     win.painter:moveto(detections.x1, detections.y1)
     win.painter:show(tracklet.item[1].objectType)

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
