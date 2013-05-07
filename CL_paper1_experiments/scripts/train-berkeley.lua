----------------------------------------------------------------------
-- E. Culurciello Fall 2012
-- Run k-means on Berkeley image and generate layers filters
-- simulate the Online Learner (OL) network as a robotic vision template
----------------------------------------------------------------------

import 'torch'
require 'image'
require 'unsup'
require 'nnx'
require 'eex'
--require 'MulAnySize'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-visualize', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 7, 'size of each input patches') -- OL: 7x7
cmd:option('-nkernels1', 16, 'number of kernels 1st layer') -- OL: 16
cmd:option('-nkernels2', 128, 'number of kernels 2nd layer') -- OL: 128
cmd:option('-nkernels3', 128, 'number of kernels 3rd layer') -- OL: 128
cmd:option('-niter1', 15, 'nb of k-means iterations')
cmd:option('-niter2', 15, 'nb of k-means iterations')
cmd:option('-niter3', 15, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 10*1000, 'nb of random training samples')
cmd:option('-initstd1', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-initstd2', 0.02, 'standard deviation to generate random initial templates')
cmd:option('-initstd3', 0.01, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:text()
opt = cmd:parse(arg or {}) -- pass parameters to rest of file:

--if not qt then
--   opt.visualize = false
--end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

is = opt.inputsize
nk1 = opt.nkernels1
nk2 = opt.nkernels2
nk3 = opt.nkernels3

normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data.lua'

filename = paths.basename(opt.datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. opt.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, opt.inputsize)
--dataset:conv()

trsize = 256--dataset:size()

trainData = {
   data = torch.Tensor(trsize, 3, dataset[1][3]:size(1), dataset[1][3]:size(2)),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData.data[t][1] = dataset[t][3]
   trainData.data[t][2] = trainData.data[t][1]
   trainData.data[t][3] = trainData.data[t][1]
   xlua.progress(t, trainData:size())
end

f256S = trainData.data[{{1,256}}]
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=1, legend='Input images'}


-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('1st layer training data mean: ' .. trainMean)
print('1st layer training data standard deviation: ' .. trainStd)


----------------------------------------------------------------------
print '==> generating 1st layer filters:'
print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
data1 = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   img = math.random(1,dataset:size())
   img2 = dataset[i][3]
   x = math.random(1,dataset[1][3]:size(1)-is+1)
   y = math.random(1,dataset[1][3]:size(2)-is+1)
   randompatch = img2[{{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data1[i] = randompatch
end

-- show a few patches:
f256S = data1[{{1,256}}]:reshape(256,is,is)
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}

print '==> running k-means'
 function cb (kernels1)
    if opt.visualize then
       win1 = image.display{image=kernels1:reshape(nk1,is,is), padding=2, symmetric=true, 
       zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}
    end
end                    
kernels1 = unsup.kmeans(data1, nk1, opt.initstd1, opt.niter1, opt.batchsize, cb, true)

-- clear nan kernels if kmeans initstd is not right!
for i=1,nk1 do   
   if torch.sum(kernels1[i]-kernels1[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels1[i] = torch.zeros(kernels1[1]:size()) 
   end
 
   -- normalize kernels to 0 mean and 1 std:  
   kernels1[i]:add(-kernels1[i]:mean())
   kernels1[i]:div(kernels1[i]:std())
end

-- visualize final kernels:
image.display{image=kernels1:reshape(nk1,is,is), padding=2, symmetric=true, 
       zoom=2, win=win1, nrow=math.floor(math.sqrt(nk1)), legend='1st layer filters'}

--print('==> saving centroids to disk:')
--torch.save('berkeley56x56-1l.t7', kernels1:clone())


----------------------------------------------------------------------
print "==> loading and initialize 1 layer CL model"

o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
cvstepsize = 1
fanin = 1
poolsize = 2
l1netoutsize = torch.floor(o1size/poolsize/cvstepsize) -- attention, here there is a FRACTION number!

-- spatialSAD:
model = nn.Sequential()
model:add(nn.SpatialSADMap(nn.tables.random(3, nk1, fanin), is, is)) -- here all 3 input maps are = so random means nothing
model:add(nn.SpatialContrastiveNormalization(nk1, normkernel))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

-- spatial conv:
--model = nn.Sequential()
--model:add(nn.SpatialConvolutionMap(nn.tables.random(3, nk1, fanin), is, is))
--model:add(nn.Tanh())
--model:add(nn.SpatialLPPooling(nk1, 2, poolsize, poolsize, poolsize, poolsize)) 
--model:add(nn.SpatialSubtractiveNormalization(nk1, normkernel))

l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
l1net.modules[1].weight = kernels1:reshape(nk1, is, is)-- SpatialSAD
l1net.modules[1].bias = l1net.modules[1].bias*0

--tests:
inp = torch.Tensor(100)
for t = 1,100 do
   l1net:forward(trainData.data[t])--:double())
   inp[t] = l1net.modules[2].output:max()
end
print('MAX output after SpatialContrastNorm:', inp:mean())
--image.display{image=inp, padding=2, symmetric=true, 
--         zoom=2, nrow=8, legend='example of 1st layer output'}

--save layer:
layer1 = l1net:clone()
torch.save('berkeley-sadmap1.net', l1net.modules[1])

----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData2 = {
   data = torch.Tensor(trsize, nk1*(l1netoutsize)^2),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData2.data[t] = l1net:forward(trainData.data[t])--:double())
   xlua.progress(t, trainData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())

trainData2.data = trainData2.data:reshape(trsize, nk1, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData1 = trainData -- save original dataset
trainData = trainData2 -- relocate new dataset

-- show a few outputs:
f256S = trainData2.data[{ {1,256},1 }]
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Output 2nd layer: first 256 examples, 1st plane'}

-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('2nd layer training data mean: ' .. trainMean)
print('2nd layer training data standard deviation: ' .. trainStd)


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------



----------------------------------------------------------------------
print '==> generating 2nd layer filters:'
print '==> extracting patches'
data2 = torch.Tensor(opt.nsamples,nk1*is*is)
for i = 1,opt.nsamples do
   img = math.random(1,trainData.data:size(1))
   img2 = trainData.data[img]
   z = math.random(1,trainData.data:size(2))
   x = math.random(1,trainData.data:size(3)-is+1)
   y = math.random(1,trainData.data:size(4)-is+1)
   randompatch = img2[{ {},{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data2[i] = randompatch
end

-- show a few patches:
--f256S2 = data2[{{1,256}}]:reshape(256,is,is)
--image.display{image=f256S2, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}

--if not paths.filep('berkeley56x56-2l.t7') then
print '==> running k-means'
function cb2 (kernels2)
   if opt.visualize then
      win2 = image.display{image=kernels2:reshape(nk2,nk1,is,is)[{{},{1},{},{}}]:reshape(nk2,is,is), padding=2, symmetric=true, zoom=2, win=win2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'} -- only one plane!
   end
end                    
kernels2 = unsup.kmeans(data2, nk2, opt.initstd2, opt.niter2, opt.batchsize, cb2, true)

-- clear nan kernels if kmeans initstd is not right!
for i=1,nk2 do   
   if torch.sum(kernels2[i]-kernels2[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels2[i] = torch.zeros(kernels2[1]:size()) 
   end
   
   -- normalize kernels to 0 mean and 1 std:  
   kernels2[i]:add(-kernels2[i]:mean())
   kernels2[i]:div(kernels2[i]:std())
end

-- visualize final kernels:
image.display{image=kernels2:reshape(nk2,nk1,is,is)[{{},{1},{},{}}]:reshape(nk2,is,is), padding=2, symmetric=true, zoom=2, win=win2, nrow=math.floor(math.sqrt(nk2)), legend='2nd layer filters'} -- only one plane!

--print('==> saving centroids to disk:')
torch.save('berkeley56x56-2l.t7', kernels2:clone())

----------------------------------------------------------------------
print "==> initializing 2nd layer CL model"

o1size = trainData.data:size(3) - is + 1 -- size of spatial conv layer output
fanin = 8
cvstepsize = 1
poolsize = 2
l1netoutsize = torch.floor(o1size/poolsize/cvstepsize) -- attention, here there is a FRACTION number!

-- spatialSAD:
model = nn.Sequential()
--model:add(nn.SpatialSADMap(nn.tables.random(nk1, nk2, fanin), is, is))
model:add(nn.SpatialSAD(nk1, nk2, is, is))
model:add(nn.SpatialContrastiveNormalization(nk2, normkernel))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

-- spatial conv:
--model = nn.Sequential()
--model:add(nn.SpatialConvolutionMap(nn.tables.random(nk1, nk2, fanin), is, is))
--model:add(nn.Tanh())
--model:add(nn.SpatialLPPooling(nk2, 2, poolsize, poolsize, poolsize, poolsize)) 
--model:add(nn.SpatialSubtractiveNormalization(nk2, normkernel))

l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
-- SpatialSAD:
--l1net.modules[1].weight = kernels2:reshape(nk2,nk1,is,is)
-- spatialSADMap:
--w2=torch.cat(kernels2:resize(nk2,is,is),kernels2:resize(nk2,is,is),1)
--w22=torch.cat(w2,w2,1)
--w222=torch.cat(w22,w22,1)
--l1net.modules[1].weight = w222
-- spatialSAD better way:
--w2=torch.cat(kernels2:resize(nk2,is,is),kernels2:resize(nk2,is,is),1)
--for t1 = 1,nk2 do
--   for t2 = t1,t1+fanin do l1net.modules[1].weight[t2] = w2[t1] end
--end

-- SpatialConv:
--l1net.modules[1].weight = kernels2:reshape(nk2,1,is,is):expand(nk2,3,is,is)--:type('torch.DoubleTensor')
-- bias:
--l1net.modules[1].bias = l1net.modules[1].bias *0

--torch.save('berkeley56x56-2l-w.t7', w222:clone())

-- random filters without k-means:
-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
--l1net.modules[1].weight = kernels2:reshape(nk2, 1, is, is):expand(nk2,fanin,is,is):reshape(nk2*fanin, is, is) 

-- creating filters based on connTable:
--for i = 1, l1net.modules[1].weight:size(1) do
--   img = math.random(1,trainData.data:size(1))
--   img2 = trainData.data[img]
--   z = l1net.modules[1].connTable[i][1]
--   x = math.random(1,trainData.data:size(3)-is+1)
--   y = math.random(1,trainData.data:size(4)-is+1)
--   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
--   l1net.modules[1].weight[i] = randompatch
--end
--for i = 1, l1net.modules[1].weight:size(1) do
--   a=l1net.modules[1].connTable[i][1]
--   b=l1net.modules[1].connTable[i][2]
--   l1net.modules[1].weight[i] = kernels2:reshape(nk2,nk1,is,is)[{{b},{a},{},{}}]:reshape(is,is)
--end
-- this is for when we just use one plane in kenrnels2:
--l1net.modules[1].weight = kernels2:reshape(nk2,1,is,is):expand(nk2,fanin,is,is):reshape(nk2*fanin,is,is)

-- display filters:
--image.display{image=l1net.modules[1].weight, padding=2, symmetric=true, zoom=2, win=win2, nrow=32, legend='2nd layer filters'}

-- load kernels fully connected as done in train-cifar-CL2l-dist.lua:
l1net.modules[1]:templates(kernels2)

--bias zeroed:
l1net.modules[1].bias = l1net.modules[1].bias *0 -- no bias!

--tests:
--inp = torch.Tensor(100)
--for t = 1,100 do
--   l1net:forward(trainData.data[t])--:double())
--   inp[t] = l1net.modules[2].output:max()
--end
--print('MAX output after SpatialContrastNorm:', inp:mean())
--image.display{image=inp, padding=2, symmetric=true, 
--         zoom=2, nrow=8, legend='example of 1st layer output'}

--save layer:
layer2 = l1net:clone()
torch.save('berkeley-sadmap2.net', l1net.modules[1])


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------

----------------------------------------------------------------------
print "==> processing dataset with CL network"

trainData3 = {
   data = torch.Tensor(trsize, nk2*(l1netoutsize)^2),
--   labels = trainData.labels:clone(),
   size = function() return trsize end
}
for t = 1,trsize do
   trainData3.data[t] = l1net:forward(trainData.data[t])--:double())
   xlua.progress(t, trainData:size())
end
--trainData2.data = l1net:forward(trainData.data:double())

trainData3.data = trainData3.data:reshape(trsize, nk2, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
trainData2 = trainData -- save original dataset
trainData = trainData3 -- relocate new dataset

-- show a few outputs:
f256S = trainData3.data[{ {1,256},1 }]
image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Output 2nd layer: first 256 examples, 1st plane'}

-- verify dataset statistics:
trainMean = trainData.data:mean()
trainStd = trainData.data:std()
print('3rd layer training data mean: ' .. trainMean)
print('3rd layer training data standard deviation: ' .. trainStd)


----------------------------------------------------------------------
print '==> generating 3rd layer filters:'
print '==> extracting patches'
data3 = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   img = math.random(1,trainData.data:size(1))
   img2 = trainData.data[img]
   z = math.random(1,trainData.data:size(2))
   x = math.random(1,trainData.data:size(3)-is+1)
   y = math.random(1,trainData.data:size(4)-is+1)
   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
   -- normalize patches to 0 mean and 1 std:
   randompatch:add(-randompatch:mean())
   --randompatch:div(randompatch:std())
   data3[i] = randompatch
end

--print '==> using patches of output 2nd layer as 3rd layer features:'
--kernels3 = data3[{{1,nk3}}]:reshape(nk3,is,is):clone()
-- normalize kernels to 0 mean and 1 std:
--for i=1,nk3 do
--   kernels3[i]:add(-kernels3[i]:mean())
--   kernels3[i]:div(kernels3[i]:std())
--end

print '==> running k-means'
function cb3 (kernels3)
   if opt.visualize then
--      win3 = image.display{image=kernels3:reshape(nk3,nk2,is,is)[{{},{1},{},{}}]:reshape(nk3,is,is), padding=2, symmetric=true, zoom=2, win=win3, nrow=math.floor(math.sqrt(nk3)), legend='3rd layer filters'} -- only one plane!
      win3 = image.display{image=kernels3:reshape(nk3,is,is), padding=2, symmetric=true, zoom=2, win=win3, nrow=math.floor(math.sqrt(nk3)), legend='3rd layer filters'} -- only one plane!
   end
end                    
kernels3 = unsup.kmeans(data3, nk3, opt.initstd3, opt.niter3, opt.batchsize, cb3, true)

-- clear nan kernels if kmeans initstd is not right!
for i=1,nk3 do   
   if torch.sum(kernels3[i]-kernels3[i]) ~= 0 then 
      print('Found NaN kernels!') 
      kernels3[i] = torch.zeros(kernels3[1]:size()) 
   end
   
   -- normalize kernels to 0 mean and 1 std:  
   kernels3[i]:add(-kernels2[i]:mean())
   kernels3[i]:div(kernels2[i]:std())
end

-- visualize final kernels:
--win3 = image.display{image=kernels3:reshape(nk3,nk2,is,is)[{{},{1},{},{}}]:reshape(nk3,is,is), padding=2, symmetric=true, zoom=2, win=win3, nrow=math.floor(math.sqrt(nk3)), legend='3rd layer filters'} -- only one plane!
win3 = image.display{image=kernels3:reshape(nk3,is,is), padding=2, symmetric=true, zoom=2, win=win3, nrow=math.floor(math.sqrt(nk3)), legend='3rd layer filters'} -- only one plane!

print('==> saving centroids to disk:')
torch.save('berkeley56x56-3l.t7', kernels3:clone())


--------------------------------------------------------------
print "==> initializing 3rd layer CL model"
fanin = 16

-- spatial SAD:
model = nn.Sequential()
----model:add(nn.SpatialSAD(nk2, nk3, is, is))
model:add(nn.SpatialSADMap(nn.tables.random(nk2, nk3, fanin), is, is))
--model:add(nn.SpatialSADMap(nn.tables.oneToOne(nk3), is, is))
model:add(nn.SpatialContrastiveNormalization(nk3, normkernel))

-- spatial conv:
--model = nn.Sequential()
--model:add(nn.SpatialConvolutionMap(nn.tables.random(nk2, nk3, fanin), is, is))

l1net = model:clone()

-- initialize 1st layer parameters to learned filters (expand them for use in all channels):
-- SpatialSAD
--l1net.modules[1]:templates(kernels3:reshape(nk3, 1, is, is):expand(nk3,nk2,is,is))
-- spatialSADMap:
--w3=torch.cat(kernels3,kernels3,1)
--l1net.modules[1].weight = w3:resize(fanin*nk3,is,is)
-- another way:
--w3=torch.cat(kernels3:resize(nk3,is,is),kernels3:resize(nk3,is,is),1)
--w31=torch.cat(w3,w3,1)
--w32=torch.cat(w31,w31,1)
--w33=torch.cat(w32,w32,1)
--w34=torch.cat(w33,w33,1)
--w35=torch.cat(w34,w34,1)
--l1net.modules[1].weight = w35
--torch.save('berkeley56x56-3l-w.t7', w35:clone())

-- SpatialSAD: better way:
--w3=torch.cat(kernels3:resize(nk3,is,is),kernels3:resize(nk3,is,is),1)
--for t1 = 1,nk3 do
--   for t2 = t1,t1+fanin do l1net.modules[1].weight[t2] = w2[t1] end
--end
-- init network kernels with random patches:
--l1net.modules[1].weight = kernels3:reshape(nk3, 1, is, is):expand(nk3,fanin,is,is):reshape(nk3*fanin, is, is) 

-- creating filters based on connTable:
--for i = 1, l1net.modules[1].weight:size(1) do
--   img = math.random(1,trainData.data:size(1))
--   img2 = trainData.data[img]
--   z = l1net.modules[1].connTable[i][1]
--   x = math.random(1,trainData.data:size(3)-is+1)
--   y = math.random(1,trainData.data:size(4)-is+1)
--   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
--   -- normalize patches to 0 mean and 1 std:
--   --randompatch:add(-randompatch:mean())
--   --randompatch:div(randompatch:std())
--   l1net.modules[1].weight[i] = randompatch
--end
---- display filters:
--image.display{image=l1net.modules[1].weight, padding=2, symmetric=true, zoom=2, win=win2, nrow=128, legend='3rd layer filters'}


-- bias term
--l1net.modules[1].bias = l1net.modules[1].bias *0

--save layer:
layer3 = l1net:clone()
torch.save('berkeley-sadmap3.net', l1net.modules[1])


--tests:
inp = torch.Tensor(100)
for t = 1,100 do
   l1net:forward(trainData.data[t])--:double())
   inp[t] = l1net.modules[2].output:max()
end
print('MAX output after SpatialContrastNorm:', inp:mean())


--------------------------------------------------------------
--torch.load('c') -- break function
--------------------------------------------------------------

--
--
--
----if not paths.filep('berkeley56x56-2l.t7') then
--   print '==> running k-means'
--   function cb2 (kernels2)
--      if opt.visualize then
--         win2 = image.display{image=kernels2:reshape(nk3,is,is), padding=2, symmetric=true, zoom=2,
--            win=win2, nrow=math.floor(math.sqrt(nk3)), legend='3rd layer filters'}
--      end
--   end                    
--   kernels3 = unsup.kmeans(data2, nk3, opt.initstd2, opt.niter2, opt.batchsize, cb2, true)
----else
----   print '==> loading pre-trained k-means kernels'
----   kernels2 = torch.load('berkeley56x56-2l.t7')
----end
--
---- clear nan kernels if kmeans initstd is not right!
--for i=1,nk3 do   
--   if torch.sum(kernels3[i]-kernels3[i]) ~= 0 then 
--      print('Found NaN kernels!') 
--      kernels3[i] = torch.zeros(kernels3[1]:size()) 
--   end
--   
--   -- normalize kernels to 0 mean and 1 std:  
--   kernels2[i]:add(-kernels2[i]:mean())
--   kernels2[i]:div(kernels2[i]:std())
--end
--
--
--print('==> saving centroids to disk:')
--torch.save('berkeley56x56-3l.t7', kernels3)


