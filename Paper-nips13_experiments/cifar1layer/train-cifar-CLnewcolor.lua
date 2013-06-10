----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 1st layer generation/load and test
----------------------------------------------------------------------
-- bypass route color version: pass color info to final classifier AND net input!

require 'image'
require 'nnx'
--require 'unsup'
require 'eex'
require 'trainLayer' -- functions for Clustering Learning on video
require 'slac'
require 'online-kmeans'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-display', true, 'display kernels')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 5, 'size of each input patches')
cmd:option('-nkernels', 16, 'number of kernels to learn')
cmd:option('-niter', 15, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 10000, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:option('-savedataset', false, 'save modified dataset')
cmd:option('-classify', true, 'run classification train/test')
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

if not qt then
   opt.display = false
end

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

is = opt.inputsize
nk = opt.nkernels

opt.niter = 50
opt.slacmodel = false
opt.whitening = true
----------------------------------------------------------------------
-- loading and processing dataset:
dofile '1_data_cifar.lua'

-- input image dateaset params:
ivch = trainData.data[1]:size(1) -- channels
ivhe = trainData.data[1]:size(2) -- height
ivwi = trainData.data[1]:size(3) -- width

trainData.data = trainData.data:double()
testData.data = testData.data:double()

----------------------------------------------------------------------
----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> generating CL unsupervised network:'

-- compute network CL train time
   time = sys.clock()

----------------------------------------------------------------------
print '==> generating filters for layer 1:'
nlayer = 1
nnf1 = 1 -- number of frames from input video to use
nk1 = 32
nk = nk1
is = 5
poolsize = 2
cvstepsize = 5
normkernel = image.gaussian1D(7)
ovhe = 14  -- output video feature height
ovwi = 14  -- output video feature width


-- AND/OR model or FULL CONNECT MODEL:
-- get twice the kernels, then pick best ones:
kernels1u, counts1, M, P = trainLayer(nlayer, trainData.data, opt.nsamples, nil, 2*nk1, nnf1, is)
-- sort kernels:
_, j = torch.sort(counts,true)
kernels1 = torch.Tensor(nk1,ivch, is,is)
for i=1,nk1 do
   kernels1[i] = kernels1u[j[i]]
end
if opt.display then
  image.display{image=kernels1:reshape(kernels1:size(1),ivch,is,is), padding=2, symmetric=true, zoom=4, legend = 'Layer 1 filters'} end


   
----------------------------------------------------------------------
-- 1st layer
  
   -- Trainable Network
vnet = nn.Sequential()
vnet:add(nn.SpatialConvolution(ivch, nk1, is, is, cvstepsize,cvstepsize))
vnet:add(nn.Threshold())
vnet:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))


-- setup net/ load kernels into network:
vnet.modules[1].bias = vnet.modules[1].bias*0 -- set bias to 0!!! not needed
vnet.modules[1].weight = kernels1:reshape(nk1, ivch, is,is)


function whiten_image(data, M, P, is)   -- operation:  data = (data - M) x P
 
       -- from 2D image to a list of 1D patches
   local npatches = (32-is)+1
   local nsample = data:size(1)
   local x = torch.zeros(nsample*npatches*npatches, 3*is*is)
   local m = 1
   for i = 1, nsample do
      for j = 1, npatches-1 do
         for k = 1, npatches-1 do
             -- slice 32x32 image (ignore 1px border) into 36 patches of 5x5
             x[{m}] = data[{i,{},{j,j+is-1},{k,k+is-1}}]:reshape(3*is*is)   --{((i-1)*6+(j-1))*6+k}
             m = m+1
          end
       end
   end
   

       -- do whitening
   x:add(torch.ger(torch.ones(nsample*npatches*npatches), M:squeeze()):mul(-1))
   x = x * P
   data2 = torch.zeros(nsample,3, npatches*is, npatches*is)                                                                     
   local m =1                                                                                                                   
    -- from the list of 1D patches to 2D image                                                                                   
    for i = 1, nsample do                                                                                                        
      for j = 1, npatches-1 do                                                                                                  
        for k = 1, npatches-1 do                                                                                               
           local xi = 1 + (k-1)*is                                                                                            
           local yi = 1 + (j-1)*is                                                                                            
           data2[{i,{}, {yi,yi+is-1},{xi,xi+is-1}}] = x[{m}]:reshape(3,is,is)                                                  
           m = m+1                                                                                                             
         end                                                                                                                    
       end                                                                                                                       
     end
    return data2                                                                                                                 
end


trainData2 = torch.Tensor(trsize, nk1, ovhe, ovhe)
testData2 = torch.Tensor(tesize, nk1, ovhe, ovhe)

 step1 = 1000                                                                                                                    
  for i=1, 50000/step1 do                                                                                                         
    print '==> process train dataset throught 1st layer:'                                                                         
    print('i', i)                                                                                                                 
    data = whiten_image(trainData.data[{{(i-1)*step1+1, i*step1}, {}, {}, {}}], M, P, is)                                         
    trainData2[{{(i-1)*step1+1, i*step1}, {}, {}, {}}] = processLayer(nlayer, vnet, data, nk1, ovhe, ovwi, false)                 
    
  end
  
  
  for i=1, 2000/step1 do                                                                                                          
    print '==> process test dataset throught 1st layer:'                                                                          
  
    data = whiten_image(testData.data[{{(i-1)*step1+1, i*step1}, {}, {}, {}}], M, P, is)                                          
    testData2[{{(i-1)*step1+1, i*step1}, {}, {}, {}}] = processLayer(nlayer, vnet, data, nk1, ovhe, ovwi, false)                  
    
  end
  




mean = {}
std = {}
for i=1, nk1 do
      -- normalize each channel globally:
       mean[i] = trainData2[{ {},i,{},{} }]:mean()
       std[i] = trainData2[{ {},i,{},{} }]:std()
       trainData2[{ {},i,{},{} }]:add(-mean[i])
       trainData2[{ {},i,{},{} }]:div(std[i])
       testData2[{ {},i,{},{} }]:add(-mean[i])
       testData2[{ {},i,{},{} }]:div(std[i])
 end



-- compute network creation time time 
time = sys.clock() - time
print("<net> time to CL train network = " .. (time*1000) .. 'ms')



----------------------------------------------------------------------
print "==> creating final test dataset"

l1netoutsize = ovhe -- 2 layers:



-- color bypass: downsamples color info and pass it to final classifier:
nlayer=1
cnpoolsize=4
colornet = nn.Sequential()
colornet:add(nn.SpatialDownSampling(cnpoolsize,cnpoolsize,cnpoolsize,cnpoolsize))
cdatasize = (ivhe/cnpoolsize) -- size of the color data


-- process dataset throught net:

trainDataF = {
   data = torch.Tensor(trsize, nk*(l1netoutsize)^2+3*(cdatasize)^2),
   color = torch.Tensor(trsize,3, cdatasize,cdatasize),  -- ad bypass color info
   labels = trainData.labels:clone(),
   size = function() return trsize end
  
}

testDataF = {
   data = torch.Tensor(tesize, nk*(l1netoutsize)^2+3*(cdatasize)^2),
   color = torch.Tensor(trsize, 3, cdatasize, cdatasize),  -- ad bypass color info
   labels = testData.labels:clone(),
   size = function() return tesize end
}

--trainDataF.data = trainData3
--testDataF.data = testData3

print '==> process color info of dataset throught colornet:'
for t = 1,trsize do
   trainDataF.color[t] = colornet:forward(trainData.data[t])
   xlua.progress(t, trainData:size())
end
for t = 1,tesize do
   testDataF.color[t] = colornet:forward(testData.data[t])
   xlua.progress(t, testData:size())
end

mean = {}                                                                                                                                  
 std = {}
 print(trainDataF.color:size())
for i=1, 3 do                                                                                                                            
      -- normalize each channel globally:                                                                                                  
   mean[i] = trainDataF.color[{ {},i,{},{} }]:mean()                                                                                         
   std[i] = trainDataF.color[{ {},i,{},{} }]:std()                                                                                           
   trainDataF.color[{ {},i,{},{} }]:add(-mean[i])                                                                                            
   trainDataF.color[{ {},i,{},{} }]:div(std[i])                                                                                              
   testDataF.color[{ {},i,{},{} }]:add(-mean[i])                                                                                             
   testDataF.color[{ {},i,{},{} }]:div(std[i])                                                                                               
end



for t = 1,trsize do
   trainDataF.data[t] = torch.cat(trainData2[t]:reshape(nk*(l1netoutsize)^2), trainDataF.color[t]:reshape(3*(cdatasize)^2))
   xlua.progress(t, trainData:size())
end
for t = 1,tesize do
   testDataF.data[t] = torch.cat(testData2[t]:reshape(nk*(l1netoutsize)^2), testDataF.color[t]:reshape(3*(cdatasize)^2))
   xlua.progress(t, testData:size())
end


--trainDataF.data = trainDataF.data:reshape(trsize, nk2, l1netoutsize, l1netoutsize)
--testDataF.data = testDataF.data:reshape(tesize, nk2, l1netoutsize, l1netoutsize)

-- relocate pointers to new dataset:
--trainData1 = trainData -- save original dataset
--testData1 = testData
trainData = trainDataF -- relocate new dataset
testData = testDataF

print('trainData:size()', trainData.data:size())
----------------------------------------------------------------------


----------------------------------------------------------------------
-- classifier for train/test:
if opt.classify then
   ----------------------------------------------------------------------
   print "==> creating classifier"
   
--   opt.model = '2mlp-classifier'
--   dofile '2_model.lua' 
   
   nhiddens = 256
   outsize = 10 -- in CIFAR, SVHN datasets

   model = nn.Sequential()
   model:add(nn.Reshape(nk*l1netoutsize^2+3*cdatasize^2))
   model:add(nn.Linear(nk*l1netoutsize^2+3*cdatasize^2, nhiddens))
   model:add(nn.Threshold())
   model:add(nn.Linear(nhiddens,outsize))
   
   print "==> test network output:"
   print(model:forward(trainData.data[1]:double()))
   
   dofile '3_loss.lua' 
   dofile '4_train.lua'
   dofile '5_test.lua'
   
   ----------------------------------------------------------------------
   print "==> training classifier"
   
   while true do
      train()
      test()
   end
   
end


