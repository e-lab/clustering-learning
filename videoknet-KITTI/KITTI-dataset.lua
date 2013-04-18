----------------------------------------------------------------------
-- KITTI dataset interface
--
----------------------------------------------------------------------

require 'nnx'
require 'image'
require 'ffmpeg'
require 'xml' --https://github.com/clementfarabet/lua---xml, donwloaded and  installed with 'torch-pkg deploy'


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

dspath = '../../datasets/KITTI/2011_09_26_drive_0060/image_03/data'--/0000000000.png' -- Right images
source = ffmpeg.Video{path=dspath, width = 310, height = 94, encoding='png', fps=10, loaddump=true, load=true}

rawFrame = source:forward()
-- input video params:
ivch = rawFrame:size(1) -- channels
ivhe = rawFrame:size(2) -- height
ivwi = rawFrame:size(3) -- width
source.current = 1 -- rewind video frames


-- number of frames to process:
nfpr = 200 -- batch process size [video frames]

-- normalize and prepare dataset:
neighborhood = image.gaussian1D(9)
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

createDataBatch()
