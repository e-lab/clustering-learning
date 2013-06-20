------------------------------------------------------------------------------
-- Basic demo
------------------------------------------------------------------------------
-- Alfredo Canziani Jun 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
lapp = require 'pl.lapp'
require 'nnx'
require 'Dropout'
require 'image'
require 'ffmpeg'

-- Parsing the command line --------------------------------------------------
   print '==> Processing options'
   opt = lapp [[
   -v,--video              (string)             Use the German road sign dataset
   -w,--width              (default 256)        Width of the video
   -h,--height             (default 144)        Height of the video
   -t,--time               (default 10)         Seconds of video to process
   -k,--seek               (default 0)          Seek number of seconds
   -z,--zoom               (default 1)          Zoom for displaying purpose
]]

-- Parameters ----------------------------------------------------------------
local normalisation = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()
local channels = {'y','u','v'}

-- Main program -------------------------------------------------------------
io.write('Insert network\'s name: ')
netName = io.read()
net = torch.load(netName)
--netName = 'multinet-18Jun-17.26.net'
--net = torch.load('/Users/atcold/Dropbox/Multinet/Networks/'..netName)
torch.setdefaulttensortype(torch.typename(net.output))
net.modules[2] = nn.SpatialClassifier(net.modules[2])

if opt.video:find('jpg') or opt.video:find('png') then
   local img = image.load(opt.video)
   --img = image.scale(img, tonumber(opt.width), tonumber(opt.height))
   video = {}
   video.forward = function() return img end
   opt.width = (#img)[3]
   opt.height = (#img)[2]
else
   video = ffmpeg.Video{path=opt.video, width=opt.width, height=opt.height, length=opt.time, seek=opt.seek, encoding='jpg', delete=false}
end

while true do
   currentFrame = video:forward()
   imgYUV = image.rgb2yuv(currentFrame)
   imgSCN = torch.Tensor(imgYUV:size())
   for c,channel in ipairs(channels) do
      imgSCN[c] = normalisation:forward(imgYUV[{ {c},{},{} }])
   end

   distribution = net:forward(imgSCN)
   map = torch.Tensor(3,(#distribution)[2],(#distribution)[3])
   map[{ {1,2},{},{} }] = distribution[{ {1,2},{},{} }]
   map[3] = distribution[4]
   _,m = distribution:max(1)
   maxMap = torch.Tensor(map:size())
   maxMap[1] = m:eq(1)
   maxMap[2] = m:eq(2)
   maxMap[3] = m:eq(4)
   masc = torch.Tensor(3,opt.height,opt.width)
   image.scale(maxMap,masc,'simple')
   overlaid = masc + currentFrame
   win = image.display{image=overlaid,win=win,zoom=opt.zoom}
end
