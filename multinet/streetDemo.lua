#!/usr/bin/env torch
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
require 'PyramidPacker'

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
--net = torch.load(netName)
--netName = 'multinet-18Jun-17.26.net'
net = torch.load('/Users/atcold/Dropbox/Multinet/Networks/'..netName)
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

-- logspace(-1,0,11)
scalesR = {0.1000, 0.1259, 0.1585, 0.1995, 0.2512, 0.3162, 0.3981, 0.5012, 0.6310, 0.7943, 1.0000}
scales = {}; for i,s in ipairs(scalesR) do scales[#scalesR-i+1] = s end
packer = nn.PyramidPacker(net, scales)

while true do
   currentFrame = packer(video:forward())
   --image.display(currentFrame)
   --io.read()
   imgYUV = image.rgb2yuv(currentFrame)
   imgSCN = torch.Tensor(imgYUV:size())
   for c,channel in ipairs(channels) do
      imgSCN[c] = normalisation:forward(imgYUV[{ {c},{},{} }])
   end

   distribution = net:forward(imgSCN)
   map = torch.Tensor(3,(#distribution)[2],(#distribution)[3])
   th = -.05
   distribution[1]:apply(function(x) return (x > th) and x or -500 end)
   distribution[2]:apply(function(x) return (x > th) and x or -500 end)
   distribution[4]:apply(function(x) return (x > th) and x or -500 end)
   -- for c = 1,4 do
   --    gnuplot.figure(c)
   --    gnuplot.hist(distribution[c])
   --    gnuplot.title('Classe ' .. c)
   -- end
   map[{ {1,2},{},{} }] = distribution[{ {1,2},{},{} }]
   map[3] = distribution[4]
   --map[3] = distribution[3]
   _,m = distribution:max(1)
   maxMap = torch.Tensor(map:size())
   maxMap[1] = m:eq(1)
   maxMap[2] = m:eq(2)
   maxMap[3] = m:eq(4)
   --maxMap[3] = m:eq(3)
   masc = torch.Tensor(currentFrame:size())
   image.scale(masc,maxMap,'simple')
   --masc = image.scale(maxMap,opt.width,opt.height,'simple'):float()
   overlaid = masc + currentFrame
   win = image.display{image=overlaid,win=win,zoom=opt.zoom,min=0,max=2}
   -- win2 = image.display{image=maxMap,zoom=55,win=win2}
   -- win3 = image.display{image=masc,win=win3,zoom=opt.zoom}
   -- win4 = image.display{image=currentFrame,win=win4,zoom=opt.zoom}
   -- io.read()
   --return
end
