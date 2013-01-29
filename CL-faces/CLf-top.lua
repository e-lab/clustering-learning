----------------------------------------------------------------------
-- Clustering Learning deep conv net on a face detection dataset
-- (original: Clement Farabet, Benoit Corda  |  July  7, 2011, 12:45PM)
-- Clustering Learning CL version: Eugenio Culurciello Jan 24th 2013
----------------------------------------------------------------------

require 'image'
require 'nnx'
require 'unsup'
require 'optim'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Face Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('--save', 'face.net', 'file to save network after each epoch')
   cmd:option('--load', false, 'reload pretrained network')
   cmd:option('--seed', 0, 'use fixed seed for randomized initialization')
   cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-patches', 'all', 'nb of patches to use')
   cmd:option('-visualize', 'visualize the datasets')
   cmd:option('-dataset', '../datasets/faces_cut_yuv_32x32/','path to FACE dataset root dir')
   cmd:option('-www', 'http://data.neuflow.org/data/faces_cut_yuv_32x32.tar.gz', 
      'path to retrieve dataset online (if not available locally)')
   cmd:option('-testset', 0.2, 'percentage of samples to use for testing')
   -- CL options:
   cmd:option('-visualize', true, 'display kernels')
   cmd:option('-seed', 1, 'initial random seed')
   cmd:option('-threads', 8, 'threads')
   cmd:option('-inputsize', 5, 'size of each input patches')
   cmd:option('-nkernels', 32, 'number of kernels to learn')
   cmd:option('-niter', 15, 'nb of k-means iterations')
   cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
   cmd:option('-nsamples', 10*1000, 'nb of random training samples')
   cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
   cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
   cmd:option('-savedataset', false, 'save modified dataset')
   cmd:option('-classify', true, 'run classification train/test')

   cmd:text()
   opt = cmd:parse(arg or {})
end


torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.seed)


----------------------------------------------------------------------
-- function max
function GetMax(a)
	x,xi = torch.max(a,1)
	y,yi = torch.max(x,2) -- y = value

	x_out = yi[1][1]  -- y coord
	y_out = xi[1][x_out] -- x coord
	return y,x_out,y_out 
end


----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data_faces.lua'


----------------------------------------------------------------------
-- train CL filters layer 1:
is1 = 5
nk1 = 16
is = is1
nk = nk1
dofile 'CLf-trainCLlayer1.lua'


----------------------------------------------------------------------
-- initialize CL layer 1

o1size = trainData[1][1]:size(3) - is1 + 1 -- size of spatial conv layer output
poolsize = 2
l1netoutsize = o1size/poolsize

l1net = nn.Sequential()
l1net:add(nn.SpatialConvolution(trainData[1][1]:size(1), nk1, is1, is1))
l1net:add(nn.Tanh())
l1net:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
l1net:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

-- initialize templates:
l1net.modules[1].weight = kernels:reshape(nk, 1, is1, is1)
l1net.modules[1].bias = l1net.modules[1].bias *0

--torch.save('l1net.net', l1net)

-- test: break if it fails!
l1net:forward(trainData[1][1])


----------------------------------------------------------------------
-- processing dataset with CL model:
--trainsize = 10000--trainData:size() -- reduce the numbers if you need to go fast@
--testsize = 2000--testData:size()
trainsize = trainData:size() -- reduce the numbers if you need to go fast@
testsize = testData:size()
dofile 'CLf-process-ds1.lua'


----------------------------------------------------------------------
-- train CL filters layer 2:
is2 = 5
nk2 = 32
is = is2
nk = nk2
dofile 'CLf-trainCLlayer2.lua'


----------------------------------------------------------------------
-- initialize CL layer 2

o2size = trainData2.data[1]:size(3) - is2 + 1 -- size of spatial conv layer output
poolsize = 2
l2netoutsize = o2size/poolsize

l2net = nn.Sequential()
l2net:add(nn.SpatialConvolution(nk1, nk2, is2, is2))
l2net:add(nn.Reshape(nk2*o2size*o2size))
l2net:add(nn.Mul(nk2*o2size*o2size))
l2net:add(nn.Reshape(nk2,o2size,o2size))
l2net:add(nn.Tanh())
l2net:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
l2net:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

-- initialize templates:
l2net.modules[1].weight = kernels2:reshape(nk2,1,is2,is2):expand(nk2,nk1,is2,is2)
l2net.modules[1].bias = l2net.modules[1].bias *0
l2net.modules[3].weight = torch.ones(1)*(1/nk)
--torch.save('l1net.net', l1net)

-- test: break if it fails!
l2net:forward(trainData2.data[1])


----------------------------------------------------------------------
-- processing dataset with CL model:
dofile 'CLf-process-ds2.lua'


----------------------------------------------------------------------
-- now MAGIC: find neurons that might encode for a face and NOT bg:

-- image.display(trainData[n][1]) -- to see if face or not
-- if trainData[n][2][1] == 1 --it is a face (if bg returns -1)

-- let's average all face output and look for most active neurons:
facefound = 0
faceave = torch.zeros(#trainData3.data[1])
for i = 1, trainsize do
   if (trainData[i][2][1] == 1) then
      facefound = facefound + 1
      faceave = faceave + trainData3.data[i]
   end
end
faceavenorm = faceave/facefound
print('Number of faces found in train set:', facefound)
image.display{image=faceavenorm, padding=2, zoom=4}


-- average of input image for top face neuron:
vmav, xm, ym =  GetMax(faceavenorm:sum(1):reshape(5,5)) -- max neuron based on average of all faces
tnave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   tnave = tnave + trainData[i][1]*trainData3.data[i]:sum(1)[1][ym][xm] -- average dataset weighted by top neuron
end
image.display{image=tnave/trainsize, padding=2, zoom=4}


-- countertest: random neuron:
rndave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   rndave = rndave + trainData[i][1]*trainData3.data[i]:sum(1)[1][1][5] -- average dataset weighted by random neuron
end
image.display{image=rndave/trainsize, padding=2, zoom=4}


-- average of dataset:
dsave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   dsave = dsave + trainData[i][1] -- average of all dataset
end
image.display{image=dsave/trainsize, padding=2, zoom=4}


