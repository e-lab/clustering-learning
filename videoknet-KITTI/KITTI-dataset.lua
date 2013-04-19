----------------------------------------------------------------------
-- KITTI dataset interface
--
----------------------------------------------------------------------

require 'nnx'
require 'image'
require 'ffmpeg'
require 'xml' --https://github.com/clementfarabet/lua---xml, donwloaded and  installed with 'torch-pkg deploy'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'sys'

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

----------------------------------------------------------------------
print '==> test KITTI dataset'

----------------------------------------------------------------------
print '==> load KITTI tracklets'
trackletFile =  '../../datasets/KITTI/2011_09_26_drive_0001/tracklet_labels.xml'


Tracklet = {
   --representation an annotated object track 
	--  
	--  Tracklets are created in function parseXML and can most conveniently used as follows:
	--
	--  for trackletObj in parseXML(trackletFile):
	--    for absoluteFrameNumber, translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders in trackletObj:
	--      ... your code here ...
	--    #end: for all frames
	--  #end: for all tracklets
	--
	--  absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
	--  amtOcclusion and amtBorders could be None
	--
	--  You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int), 
	--    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
	--    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
	--    can be None if the xml file did not include these fields in poses
	
  objectType = '',
  size = 0, --None  # len-3 float array: (height, width, length)
  firstFrame = 0, --None
  trans = 0, --None   # n x 3 float array (x,y,z)
  --rots = 0, --None    # n x 3 float array (x,y,z)
  --states = 0, --  # len-n uint8 array of states
  --occs = 0, --    # n x 2 uint8 array  (occlusion, occlusion_kf)
  --truncs = 0, --  # len-n uint8 array of truncation
  --amtOccs = 0, --    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
  --amtBorders = 0, --None    # None (n x 3) float array  (amt_border_l / _r / _kf)
  nFrames = 0 --None
  --size = function() return xvalsize end
}
-- constructor function:
function Tracklet:new(o)
	o = o or {}   -- create object if user does not provide one
	setmetatable(o, self)
	self.__index = self
	return o
end
	

function parseXML(trackletFile)
	--param trackletFile: name of a tracklet xml file
	--returns: list of Tracklet objects read from xml file
	--return string.match(xml.str(a[1]), '%d+') -- to get a value, but not needed!
	
	local labels = xml.load(trackletFile)
	local a = labels:find('tracklets')
	local tracklets = {}
	for i= 1, tonumber(a[1][1]) do -- for each tracklet
		tracklets[i] = Tracklet:new()
		tracklets[i].objectType = a[i+2][1][1]
		tracklets[i].firstFrame = a[i+2][5][1]
		tracklets[i].size = torch.Tensor({a[i+2][2][1], a[i+2][3][1], a[i+2][4][1]}) -- h,w,l
		tracklets[i].trans = torch.Tensor(tonumber(a[i+2][6][1][1]), 3)
		for j = 1, tonumber(a[i+2][6][1][1]) do -- for each frame/pose
			tracklets[i].trans[j] =  torch.Tensor({a[i+2][6][3][1][1], 
				a[i+2][6][3][2][1], a[i+2][6][3][3][1]}) -- x,y,z
		end
		tracklets[i].nFrames = a[i+2][6][1][1] -- number of frames/poses per tracklet
	end
	return tracklets
end

tracklets = parseXML(trackletFile)

----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

--dspathL = '../datasets/KITTI/2011_09_26_drive_0001/image_02/data/' -- Left images
dspath = '../../datasets/KITTI/2011_09_26_drive_0001/image_03/data/' -- Right images
--sourceL = ffmpeg.Video{path=dspathL, width = 310, height = 94, encoding='png', fps=10, loaddump=true, load=false}
--source = ffmpeg.Video{path=dspath, width = 310, height = 94, encoding='png', fps=10, loaddump=true, load=true}
imgi=0 -- pointer to image
rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi)..'.png'))

-- input video params:
ivch = rawFrame:size(1) -- channels
ivhe = rawFrame:size(2) -- height
ivwi = rawFrame:size(3) -- width
--source.current = 1 -- rewind video frames

-- test: detect = {x=100, y=20, w=20, h=30}
-- process tracklets into image bounding boxes:
win = image.display(rawFrame)
videoframes = #sys.dir(dspath)-2 -- #sys.dir(dspath) == total number of frames in video dump (minum . and ..)
for imgi = 1,videoframes do
	rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi-1)..'.png'))
	image.display{image=rawFrame, win=win}
	
	-- get bounding boxes from tracklets:
	detections = {}
	for i,res in ipairs(tracklets) do
		if imgi >= tonumber(tracklets[i].firstFrame) and 
				imgi <= tonumber(tracklets[i].firstFrame + tracklets[i].nFrames) then
			local x = tracklets[i].trans[1][1]
			local y = tracklets[i].trans[1][2]
			local w = tracklets[i].size[2]
			local h = tracklets[i].size[1]
			detections[i] = {x=x, y=y, w=w, h=h, obj_type = tracklets[i].objectType}
		end
	end

	-- paint bounding boxes:	
	for i,detect in ipairs(detections) do
      win.painter:setcolor(1,0,0)
      win.painter:rectangle(detect.x, detect.y, detect.w, detect.h)
      win.painter:stroke()
      win.painter:setfont(qt.QFont{serif=false,italic=false,size=16})
      win.painter:moveto(detect.x, detect.y-1)
      win.painter:show(detect.obj_type)
   end
end


---- number of frames to process:
--nfpr = 200 -- batch process size [video frames]
--
---- normalize and prepare dataset:
--neighborhood = image.gaussian1D(9)
--normalization = nn.SpatialContrastiveNormalization(ivch, neighborhood, 1e-3)
--
--function createDataBatch()
--   trainData = torch.Tensor(nfpr,ivch,ivhe,ivwi)
--   for i = 1, nfpr do -- just get a few frames to begin with
--      procFrame = normalization:forward(rawFrame) -- full LCN!
--      trainData[i] = procFrame
--      rawFrame = source:forward()
--      -- do a live display
--      winm = image.display{image=procFrame, win=winm}
--   end
--   return trainData
--end
--
--createDataBatch()
