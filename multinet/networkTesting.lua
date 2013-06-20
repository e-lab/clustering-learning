--------------------------------------------------------------------------------
-- This routine runs some tests on the saved <multinet> network
--------------------------------------------------------------------------------
-- Alfredo Canziani, June 2013
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'image'
require 'nnx'
require 'eex'
require 'ffmpeg'
require 'sys'
require 'Dropout'

-- Parameters ------------------------------------------------------------------
zoom = 2

-- Main program ----------------------------------------------------------------

--------------------------------------------------------------------------------
-- Gazing at the datasets ------------------------------------------------------
--------------------------------------------------------------------------------
train = torch.load('scratch/stopPedBckCarEQUAL-images/train.t7')
require 'sys'
win = image.display{image=train.data[1],win=win,zoom=5}
io.write('Position the window where desired and press <Enter>...')
io.read()
for i = 1,train.size(),2 do
   image.display{image=train.data[i],legend=i..' sample',win=win,zoom=5}
   sys.sleep(.2)
end
--------------------------------------------------------------------------------
-- Test on test dataset samples ------------------------------------------------
--------------------------------------------------------------------------------
--[[ Loading the network and making the classifier testable on large images
model = torch.load('results/multinet.net') -- loading
torch.setdefaulttensortype(torch.typename(model.output)) -- kinda bug
model.modules[2] = nn.SpatialClassifier(model.modules[2]) -- classif. alteration

-- Loading testing dataset
test = torch.load('scratch/stopPedBckCarEQUAL/test.t7')

idx = {}
idx.pds = {   1,  563}
idx.stp = { 564,  775}
idx.bgr = { 776, 1609}
idx.car = {1610, 1788}

-- Testing the testing dataset
pds = test.data[math.random(idx.pds[1],idx.pds[2])]
stp = test.data[math.random(idx.stp[1],idx.stp[2])]
car = test.data[math.random(idx.car[1],idx.car[2])]
img = torch.cat(torch.cat(pds,stp,3),car,3)

image.display{image=img,zoom=zoom}
output = model:forward(img)
output[3] = output[4]
image.display{image=output[{ {1,3},{},{} }],zoom=zoom*5.75}]]

--------------------------------------------------------------------------------
--[[ Another test on test dataset samples ----------------------------------------
--------------------------------------------------------------------------------
model = torch.load('results/multinet.net') -- loading
torch.setdefaulttensortype(torch.typename(model.output)) -- kinda bug
model.modules[2] = nn.SpatialClassifier(model.modules[2]) -- classif. alteration

imgOrg = image.load('testImg.png')
image.display{image=imgOrg,legend='Original image'}

-- Some preprocessing
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()
img = imgOrg:clone()
img[{ {1},{},{} }] = normalization:forward(img[{ {1},{},{} }])
image.display{image=img,legend='SCN-Y'}
for i = 2,3 do
   mean = img[{ i,{},{} }]:mean()
   std = img[{ i,{},{} }]:std()
   img[{ i,{},{} }]:add(-mean)
   img[{ i,{},{} }]:div(std)
end
image.display{image=img,legend='SCN-Y & UV-normalised'}

output = model:forward(img)
image.display{image=output,zoom=15,padding=2,legend='pedestrian - stop - background - car'}
map = torch.Tensor(3,(#output)[2],(#output)[3])
map[{ {1,2},{},{} }] = output[{ {1,2},{},{} }]
map[3] = output[4]
image.display{image=map,zoom=15,legend='R: pedestrian, G: stop, B: car'}
masc = imgOrg:clone()
masc = image.scale(map,masc,'simple')]]


--------------------------------------------------------------------------------
--[[ Test on test dataset samples ------------------------------------------------
--------------------------------------------------------------------------------
-- Loading the network and making the classifier testable on large images
model = torch.load('results/multinet.net') -- loading
torch.setdefaulttensortype(torch.typename(model.output)) -- kinda bug
model.modules[2] = nn.SpatialClassifier(model.modules[2]) -- classif. alteration]]

--[[ Loading testing dataset
test = torch.load('test.t7')]]

--[[ Going throgh the whole testing dataset
for i = 1, test.size(), 10 do
win2 = image.display{image=test.data[i],zoom=10,win=win2,legend=i}
io.read()
end]]

--[[ Testing the testing dataset
img1 = test.data[400]
img2 = test.data[600]
img3 = test.data[800]
img = torch.cat(torch.cat(img1,img2,3),img3,3)

image.display{image=img,zoom=zoom}
image.display{image=model:forward(img),zoom=zoom*5.75}

img1 = test.data[405]
img2 = test.data[605]
img3 = test.data[805]
imgA = torch.cat(torch.cat(img1,img2,3),img3,3)
img = torch.cat(img,imgA,2)

image.display{image=img,zoom=zoom}
image.display{image=model:forward(img),zoom=zoom*5.75}

img1 = test.data[410]
img2 = test.data[610]
img3 = test.data[810]
imgB = torch.cat(torch.cat(img1,img2,3),img3,3)
img = torch.cat(img,imgB,2)

image.display{image=img,zoom=zoom}
image.display{image=model:forward(img),zoom=zoom*5.75}]]

--[[ Loading a real image
cross = image.load('test/cross.jpg')
image.display(cross)
cross = image.rgb2yuv(image.scale(cross,(#cross)[3]*.8,(#cross)[2]*.8))
image.display(cross)

-- Some preprocessing
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()
cross[{ {1},{},{} }] = normalization:forward(cross[{ {1},{},{} }])
image.display(cross)
for i = 2,3 do
   mean = cross[{ i,{},{} }]:mean()
   std = cross[{ i,{},{} }]:std()
   cross[{ i,{},{} }]:add(-mean)
   cross[{ i,{},{} }]:div(std)
end

image.display(cross)
image.display{image=model:forward(cross),zoom=5.75}

-- Some competitive algorithm
netOut = model:forward(cross)
proc1 = netOut:clone()
proc1[1] = proc1[1] - netOut[2] - netOut[3]
proc1[2] = proc1[2] - netOut[1] - netOut[3]
proc1[3] = proc1[3] - netOut[1] - netOut[2]
proc2 = proc1:clone()

th = 28
proc2 = proc1:clone()
proc2:apply(function(x) return (x > th) and x or 0 end)
win = image.display{image=proc2,zoom=10,win=win,legend=tostring(th)}

proc3 = proc1
proc3:apply(function(x) return math.exp(x) end)
image.display{image=proc3[1],zoom=10}
image.display{image=proc3[2],zoom=10}
image.display{image=proc3[3],zoom=10}]]

--------------------------------------------------------------------------------
--[[ Test on videos --------------------------------------------------------------
--------------------------------------------------------------------------------
ls = eex.ls
path = eex.datasetsPath()
videoPath = path .. 'videos/intersection.mp4'
-- Aspect ratio = 16/9 !
w = {16*16, 1280} -- they have to be a multiple of 16 for the fovea
h = { 9*16, 720 } -- the second value is the actual resolution of the input video
io.write(sys.COLORS.red .. 'Do you want to dump the video? [y/n] ')
if io.read() == 'y' then
   io.write('The minimum resolution is 256x512. You can scale it up to 1280x720.\n' ..
            'Introduce scale factor (1,2,...): ')
   local s = io.read()
   print 'Please wait...'
   video = ffmpeg.Video{path=videoPath, width=s*w[1], height=s*h[1], length=180, encoding='jpg', delete=false}
   print 'Done :)'
   return
end

io.write(sys.COLORS.blue .. 'Choose scale factor (1,2,...): ')
local s = io.read()
imgList = ls('scratch/intersection.mp4_10fps_' ..w[1]*s ..'x'..h[1]*s..'_180s_c0_sk0_jpg/*')

multi = torch.load('results/multinet.net') -- loading
torch.setdefaulttensortype(torch.typename(multi.output)) -- kinda bug
multi.modules[2] = nn.SpatialClassifier(multi.modules[2]) -- classif. alteration

sift = torch.load('scratch/siftflow.net') -- loading
sift.modules[2] = nn.SpatialClassifier(sift.modules[2]) -- classif. alteration

neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-3):float()
winMulti, winSift = nil, nil -- disataching from previous session
print('Press <Enter> to advance to the next frame, <Ctrl>-<c>-<Enter> to terminate')
for i,imgName in ipairs(imgList) do
   io.write('Current frame: ' .. i)
   img = image.load(imgName)
   winRgb = image.display{image=img,legend='RGB frame #' .. tostring(i),win=winRgb}
   winSift  = image.display{image=sift :forward(img),win=winSift,zoom=2,legend='Sift'}

   img = image.rgb2yuv(img)
   -- Some preprocessing
   img[{ {1},{},{} }] = normalization:forward(img[{ {1},{},{} }])
   for i = 2,3 do
      mean = img[{ i,{},{} }]:mean()
      std = img[{ i,{},{} }]:std()
      img[{ i,{},{} }]:add(-mean)
      img[{ i,{},{} }]:div(std)
   end
   winYuv = image.display{image=img,legend='YUV frame #' .. tostring(i),win=winYuv}

   winMulti = image.display{image=multi:forward(img),win=winMulti,zoom=2,nrow=2,legend='Multinet'}
   io.read()
end]]
