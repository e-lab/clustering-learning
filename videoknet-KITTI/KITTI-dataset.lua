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

----------------------------------------------------------------------
print '==> test KITTI dataset'

----------------------------------------------------------------------
print '==> load KITTI tracklets'
labels = xml.load('../datasets/KITTI/2011_09_26_drive_0001/tracklet_labels.xml')

a = labels:find('tracklets')
return string.find(xml.str(a[1]), '%d')


function parseXML(trackletFile)
	tracklets = {}
	
return tracklets



def parseXML(trackletFile):
  r""" parse tracklet xml file and convert results to list of Tracklet objects
  
  :param trackletFile: name of a tracklet xml file
  :returns: list of Tracklet objects read from xml file
  """

  # convert tracklet XML data to a tree structure
  eTree = ElementTree()
  print 'parsing tracklet file', trackletFile
  with open(trackletFile) as f:
    eTree.parse(f)

  # now convert output to list of Tracklet objects
  trackletsElem = eTree.find('tracklets')
  tracklets = []
  trackletIdx = -1
  nTracklets = None
  for trackletElem in trackletsElem:
    #print 'track:', trackletElem.tag
    if trackletElem.tag == 'count':
      nTracklets = int(trackletElem.text)
    elif trackletElem.tag == 'item_version':
      pass
    elif trackletElem.tag == 'item':
      trackletIdx += 1
      #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
      # a tracklet
      newTrack = Tracklet()
      isFinished = False
      hasAmt = False
      frameIdx = None
      for info in trackletElem:
        #print 'trackInfo:', info.tag
        if isFinished:
          raise ValueError('more info on element after finished!')
        if info.tag == 'objectType':
          newTrack.objectType = info.text
        elif info.tag == 'h':
          newTrack.size[0] = float(info.text)
        elif info.tag == 'w':
          newTrack.size[1] = float(info.text)
        elif info.tag == 'l':
          newTrack.size[2] = float(info.text)
        elif info.tag == 'first_frame':
          newTrack.firstFrame = int(info.text)
        elif info.tag == 'poses':
          # this info is the possibly long list of poses
          for pose in info:
            #print 'trackInfoPose:', pose.tag
            if pose.tag == 'count':   # this should come before the others
              if newTrack.nFrames is not None:
                raise ValueError('there are several pose lists for a single track!')
              elif frameIdx is not None:
                raise ValueError('?!')
              newTrack.nFrames = int(pose.text)
              newTrack.trans  = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.rots   = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.occs   = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
              newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
              newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              frameIdx = 0
            elif pose.tag == 'item_version':
              pass
            elif pose.tag == 'item':
              # pose in one frame
              if frameIdx is None:
                raise ValueError('pose item came before number of poses!')
              for poseInfo in pose:
                #print 'trackInfoPoseInfo:', poseInfo.tag
                if poseInfo.tag == 'tx':
                  newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ty':
                  newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'tz':
                  newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'rx':
                  newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ry':
                  newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'rz':
                  newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'state':
                  newTrack.states[frameIdx] = int(poseInfo.text)
                elif poseInfo.tag == 'occlusion':
                  newTrack.occs[frameIdx, 0] = int(poseInfo.text)
                elif poseInfo.tag == 'occlusion_kf':
                  newTrack.occs[frameIdx, 1] = int(poseInfo.text)
                elif poseInfo.tag == 'truncation':
                  newTrack.truncs[frameIdx] = int(poseInfo.text)
                elif poseInfo.tag == 'amt_occlusion':
                  newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_occlusion_kf':
                  newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_l':
                  newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_r':
                  newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_kf':
                  newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                  hasAmt = True
                else:
                  raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
              frameIdx += 1
            else:
              raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
        elif info.tag == 'finished':
          isFinished = True
          if not hasAmt:
            newTrack.amtOccs = None
            newTrack.amtBorders = None
          tracklets.append(newTrack)
        else:
          raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
    else:
      raise ValueError('unexpected tracklet info')
  #end: for tracklet list items

  return tracklets
#end: function parseXML



----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

--dspathL = '../datasets/KITTI/2011_09_26_drive_0001/image_02/data' -- Left images
dspath = '../datasets/KITTI/2011_09_26_drive_0001/image_03/data'--/0000000000.png' -- Right images
--sourceL = ffmpeg.Video{path=dspathL, width = 310, height = 94, encoding='png', fps=10, loaddump=true, load=false}
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
