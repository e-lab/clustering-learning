-- Libraries ------------------------------------------------------------------
require 'sys'
require 'cunn'
require 'pl'
require 'nnx'

-- Parsing the command-line ---------------------------------------------------
opt = lapp [[
--GpuId (default 1) Choose which GPU to use
--nbOfIterations (default 100) Number of iteration over to average
--task (string) Choose which network to profile {CNN|classifier|combined}
]]

-- Setting the environment ----------------------------------------------------
cutorch.setDevice(opt.GpuId)
print('GpuId: ', cutorch.getDevice())

-- Parameters -----------------------------------------------------------------
batchSize = 128
imgSize   = 46
nk     = {16,64,256}      -- nb of features
is     = {7,5,5}          -- size of kernels
ss     = {2,2,4}          -- size of subsamplers (strides)
hidden = 1024             -- nb of hidden features for top perceptron (0=linear classifier)
clsfNk = {nk[3], hidden} -- dimensions for top perceptron
ivch   = 3
nbClasses = 34

-- Network --------------------------------------------------------------------
print 'Generating CNN network'
local CNN = nn.Sequential()

-- CUDA needs the batch dimension to be the inner most (=> transposition)
CNN:add(nn.Transpose({1,4},{1,3},{1,2}))

-- stage 1: conv+max
CNN:add(nn.SpatialConvolutionCUDA(ivch, nk[1], is[1], is[1]))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPoolingCUDA(ss[1], ss[1], ss[1], ss[1]))

-- stage 2: conv+max
CNN:add(nn.SpatialConvolutionCUDA(nk[1], nk[2], is[2], is[2]))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPoolingCUDA(ss[2], ss[2], ss[2], ss[2]))

-- stage 3: conv+max
CNN:add(nn.SpatialConvolutionCUDA(nk[2], nk[3], is[3], is[3]))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPoolingCUDA(ss[3], ss[3], ss[3], ss[3]))

-- CUDA reverse the transposition for integrability
CNN:add(nn.Transpose({1,4},{2,4},{3,4}))

-- CNN:cuda()


print 'Generating classifier'
local classifier = nn.Sequential()

classifier:add(nn.Reshape(clsfNk[1]))
classifier:add(nn.Linear(clsfNk[1],clsfNk[2]))
classifier:add(nn.Threshold())
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(clsfNk[2],nbClasses))

-- final stage: log probabilities
classifier:add(nn.LogSoftMax())

-- classifier:cuda()

-- putting network together: a <Sequential> of <Sequential>s
--    <model>
--       |___<CNN>
--       |___<classifier>

model = nn.Sequential()

if (opt.task ~= 'classifier') then
   model:add(CNN)
end
if (opt.task ~= 'CNN') then
   model:add(classifier)
end

model:cuda()

-- Training -------------------------------------------------------------------
print('Batch size: ', batchSize)
if (opt.task == 'classifier') then
   dataset = torch.randn(batchSize, clsfNk[1], 1, 1):float()
else
   dataset = torch.randn(batchSize, ivch, imgSize, imgSize):float()
end
input = dataset:cuda()

-- benchmark:
nbOfIterations = opt.nbOfIterations
print('Running ' .. nbOfIterations .. ' iterations and averaging out')

print(' + Feed forward')
sys.tic()

for t = 1,nbOfIterations do
   model:forward(input)
end
cutorch.synchronize()
t = sys.toc()/nbOfIterations

-- result:
--print('Number of ops in GOPs:', ops/1e9)
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

print(' + Feed forward, back prop and acc')
sys.tic()

for t = 1,nbOfIterations do
   model:forward(input)
   model:backward(input, model.output)
end
cutorch.synchronize()
t = sys.toc()/nbOfIterations

-- result:
--print('Number of ops in GOPs:', ops/1e9)
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

print(' + Feed forward + sending data')
sys.tic()

for t = 1,nbOfIterations do
   for i = 1, batchSize do
      input[i] = dataset[i]
   end
   model:forward(input)
end
cutorch.synchronize()
t = sys.toc()/nbOfIterations

-- result:
--print('Number of ops in GOPs:', ops/1e9)
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

print(' + Feed forward, back prop and acc + sending data')
sys.tic()

for t = 1,nbOfIterations do
   for i = 1, batchSize do
      input[i] = dataset[i]
   end
   model:forward(input)
   model:backward(input, model.output)
end
cutorch.synchronize()
t = sys.toc()/nbOfIterations

-- result:
--print('Number of ops in GOPs:', ops/1e9)
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

print()
print('Loading <siftflow> dataset')
require 'load-siftflow'

print 'Allocating minibatch memory'
x = torch.Tensor(batchSize,3,46,46):cuda()
yt = torch.Tensor(batchSize):cuda()

print()
print('Running one epoch (the whole dataset): ' .. math.floor(trainData:size()/batchSize) .. ' batches of ' .. batchSize .. ' images each')
print(' + Feed forward, back prop and acc + sending data')
sys.tic()

-- shuffle at each epoch
local shuffle = torch.randperm(trainData:size())

for t = 1,trainData:size(),batchSize do
   collectgarbage()

   -- batch fits?
   if (t + batchSize - 1) > trainData:size() then
      break
   end

   -- create mini batch
   local idx = 1
   for i = t,t+batchSize-1 do
      local a = trainData[idx]
      x[idx] = (a[1][{ {},{a[4]-23+1,a[4]+23},{a[3]-23+1,a[3]+23} }])
      yt[idx] = a[2] * l
      idx = idx + 1
   end

   y = model:forward(x)
   model:backward(x,y)
end

cutorch.synchronize()
t = sys.toc()/math.floor(trainData:size()/batchSize)
tot = t*math.floor(trainData:size()/batchSize) -- total time in s

-- result:
print('    - Time taken for 1 epoch (whole dataset): ' .. math.floor(tot/60) .. "'" .. math.floor(tot%60+.5) .. '"')
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

print(' + Accessing data from dataSetLabelMe')
sys.tic()

for t = 1,trainData:size(),batchSize do
   collectgarbage()

   -- batch fits?
   if (t + batchSize - 1) > trainData:size() then
      break
   end

   -- create mini batch
   local idx = 1
   for i = t,t+batchSize-1 do
      local a = trainData[idx]
   end

end

t = sys.toc()/math.floor(trainData:size()/batchSize)
tot = t*math.floor(trainData:size()/batchSize) -- total time in s

-- result:
print('    - Time taken for 1 epoch (whole dataset): ' .. math.floor(tot/60) .. "'" .. math.floor(tot%60+.5) .. '"')
print('    - Average time taken for 1 batch [ms]: ', t*1000)
print('    - Average time taken for 1 computation [ms]: ', t/batchSize*1000)

