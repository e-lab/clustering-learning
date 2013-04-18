----------------------------------------------------------------------
-- Run k-means on TLD dataset to extract motion filter
-- 1st layer generation of motion filters on multiple frames
-- December 20th 2012
-- E. Culurciello
----------------------------------------------------------------------

import 'torch'
require 'nnx'
require 'image'
require 'unsup'
require 'ffmpeg'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches')
cmd:option('-nkernels', 256, 'number of kernels to learn')
cmd:option('-niter', 15, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 100*1000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:option('-savedataset', false, 'save modified dataset')
cmd:option('-classify', true, 'run classification train/test')
cmd:option('-nnframes', 4, 'nb of frames uses for temporal learning of features')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', true, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to training files:

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

is = opt.inputsize
nk = opt.nkernels
nnf = opt.nnframes

----------------------------------------------------------------------
print '==> loading and processing (local-contrast-normalization) of dataset'

dspath = '/Users/eugenioculurciello/Pictures/2013/1-13-13/VID_20130105_111419.mp4'
source = ffmpeg.Video{path=dspath, encoding='png',loaddump=false, load=false}
--dspath = '../datasets/TLD/06_car'
--source = ffmpeg.Video{path=dspath, encoding='jpg',loaddump=true, load=false}
rawFrame = source:forward()

neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(rawFrame:size(1), neighborhood, 1e-3)
trainData = torch.Tensor(100,3,240,320)
for i = 1,100 do -- just get a few frames to begin with
   procFrame = normalization:forward(rawFrame) -- full LCN!
   trainData[i] = procFrame
   rawFrame = source:forward()
end

----------------------------------------------------------------------
print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data = torch.Tensor(opt.nsamples,nnf*is*is) -- need to learn volumetric filters on multiple frames!
for i = 1,opt.nsamples do
   fimg = math.random(1,trainData:size(1)-1)
   --for j = 1,nnf
      imgP = trainData[fimg] -- previous frame
      imgN = trainData[fimg+1] -- next frame
   --end
   z = math.random(1,trainData:size(2))
   x = math.random(1,trainData:size(4)-is+1)
   y = math.random(1,trainData:size(3)-is+1)
   --local patches = torch.zeros(nnf*is*is)
   --for j = 1,nnf
      randompatchP = imgP[{ {z},{y,y+is-1},{x,x+is-1} }]
      randompatchN = imgN[{ {z},{y,y+is-1},{x,x+is-1} }]
   --end
   -- normalize patches to 0 mean:
   randompatchP:add(-randompatchP:mean())
   randompatchN:add(-randompatchN:mean())
   data[i] = torch.cat(randompatchP,randompatchN)
   --data[i][1] = randompatchP
   --data[i][2] = randompatchN
end

-- show a few patches:
if opt.visualize then
   f256S = data[{{1,256}}]:reshape(256,is,nnf*is)
   image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches prev frame'}
end

--if not paths.filep('cifar10-1l.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk,is,nnf*is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='1st layer filters'}
      end
   end                    
   kernels = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true)
   print('==> saving centroids to disk:')
   torch.save('volumetric.t7', kernels)


for i=1,nk do
   -- normalize kernels to 0 mean and 1 std:
   kernels[i]:add(-kernels[i]:mean())
   kernels[i]:div(kernels[i]:std())

   -- clear nan kernels   
   if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels[i] = torch.zeros(kernels[1]:size()) 
   end
end

print '==> verify filters statistics'
print('filters max mean: ' .. kernels:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels:std(2):abs():max())

out1 = image.display{image=kernels:reshape(nk,is,nnf*is)[{{1,nk},{1,is},{1,is}}], padding=2, symmetric=true, zoom=2, offscreen=true, nrow = 16}
out2 = image.display{image=kernels:reshape(nk,is,nnf*is)[{{1,nk},{1,is},{is+1,nnf*is}}], padding=2, symmetric=true, zoom=2, offscreen=true, nrow = 16}
--image1 = kernels:reshape(nk,is,2*is)[{{1,nk},{1,is},{1,is}}]
--image2 = kernels:reshape(nk,is,2*is)[{{1,nk},{1,is},{is+1,2*is}}]
outima1 = out1.image(out1)
outima1.save(outima1,'volumetric1.jpg')
outima2 = out2.image(out2)
outima2.save(outima2,'volumetric2.jpg')

-- use imagemagick (brew package) to convert into a Animated GIF file:
os.execute('convert -delay 20 -loop 0 volumetric*.jpg volumetric.gif')

