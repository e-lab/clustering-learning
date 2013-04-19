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
			tracklets[i].trans[j] =  torch.Tensor({a[i+2][6][2+j][1][1], 
				a[i+2][6][2+j][2][1], a[i+2][6][2+j][3][1]}) -- x,y,z
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
videoframes = #sys.dir(dspath)-2 --total number of frames in video dump - (files: . and ..)
for imgi = 1,videoframes do
	rawFrame = image.loadPNG(tostring(dspath..string.format("%010u", imgi-1)..'.png'))
	image.display{image=rawFrame, win=win}
	
	-- get bounding boxes from tracklets:
	detections = {}
	for i,res in ipairs(tracklets) do
		local t1f = tracklets[i].firstFrame
		local tnf = tracklets[i].nFrames
		if imgi-1 >= tonumber(t1f) and  imgi-1 < tonumber(t1f + tnf) then
			local x = torch.abs( tracklets[i].trans[imgi-t1f][1])
			local y = torch.abs( tracklets[i].trans[imgi-t1f][2])
			local w = torch.abs( tracklets[i].size[2])
			local h = torch.abs( tracklets[i].size[1])		
		
			-- convert coordinates:
			-- In order to transform a homogeneous point X = [x y z 1]' from the velodyne
			-- coordinate system to a homogeneous point Y = [u v 1]' on image plane of
			-- camera xx, the following transformation has to be applied:
			-- Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X
			-- with:
			-- P_rect_xx (3x4):         rectfied cam 0 coordinates -> image plane
			-- R_rect_00 (4x4):         cam 0 coordinates -> rectified cam 0 coord.
			-- (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
			-- Note that the (4x4) matrices above are padded with zeros and:
			-- R_rect_00(4,4) = (R|T)_velo_to_cam(4,4) = (R|T)_imu_to_velo(4,4) = 1
		
			--R_rect_00 = torch.Tensor( {
--				{9.999239e-01, 9.837760e-03, -7.445048e-03, 1}, 
--				{-9.869795e-03, 9.999421e-01, -4.278459e-03, 1},
--				{7.402527e-03, 4.351614e-03, 9.999631e-01, 1},
--				{1,1,1,1} } )
--			P_rect_02 = torch.Tensor( {
--				{7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01}, 
--				{0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01}, 
--				{0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03} } )
--			T_velo_to_cam = torch.Tensor( {-4.069766e-03, -7.631618e-02, -2.717806e-01, 1} )
--
--			x,y,z = P_rect_02 * R_rect_00 * T_velo_to_cam * {x,y,0}
		
			--print(x,y)
		
			x=x*12 y=y*22 w=w*30 h=h*20
		
			table.insert(detections, {x=x, y=y, w=w, h=h, obj_type = tracklets[i].objectType} )
		end
	end
	print(detections)
	
	-- paint bounding boxes:	
	for i,detect in ipairs(detections) do
      win.painter:setcolor(1,0,0)
      win.painter:rectangle(detect.x, detect.y, detect.w, detect.h)
      win.painter:stroke()
      win.painter:setfont(qt.QFont{serif=false,italic=false,size=16})
      win.painter:moveto(detect.x, detect.y-1)
      win.painter:show(detect.obj_type)
   end
   
   sys.sleep(0.5) 
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
