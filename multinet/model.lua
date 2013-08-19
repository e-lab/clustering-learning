----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout'

--[[opt={}
opt.dropout = .5
classes = {1,2,3,4}]]

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end


----------------------------------------------------------------------
-- define network to train

print '==> Defining network parameters:'

local nk1,nk2,nk3   = 16,64,256--32,64    -- nb of features
local is1,is2,is3   = 7,5,5      -- size of kernels
local ss1,ss2,ss3   = 2,2,4         -- size of subsamplers (strides)
local hiddens       = 1024--64            -- nb of hidden features for top perceptron (0=linear classifier)
local cl_nk1,cl_nk2 = nk3, hiddens  -- dimensions for top perceptron
local ivch          = 3

--[[ Old net
nk0,nk1,nk2,nk3 = 3,16,128,256      -- nb of features
is0,is1,is2,is3 = 15,7,7,7          -- size of kernels
ss1,ss2         = 2,2               -- size of subsamplers (strides)
hiddens         = 512               -- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2   = nk3, hiddens      -- dimensions for top perceptron
ivch            = 3]]

-- dropout?
local dropout = nn.Dropout(opt.dropout)

----------------------------------------------------------------------
if opt.siftflow then
   local filterSize = 15
   local planes = 3
   local normthres = 1e-1

   -- Preprocessor (normalizer)
   preproc = nn.Sequential()
   if opt.preproc == 'norm(rgb)' then
      preproc:add(nn.SpatialContrastiveNormalization(planes, image.gaussian1D(filterSize), normthres))
   elseif opt.preproc == 'norm(yuv)' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      preproc:add(nn.SpatialContrastiveNormalization(planes, image.gaussian1D(filterSize), normthres))
   elseif opt.preproc == 'norm(y)+norm(u)+norm(v)' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      do
         normer = nn.ConcatTable()
         for i = 1,3 do
            local n = nn.Sequential()
            n:add(nn.Narrow(1,i,1))
            n:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(filterSize), normthres))
            normer:add(n)
         end
      end
      preproc:add(normer)
      preproc:add(nn.JoinTable(1))
   elseif opt.preproc == 'norm(y)+uv' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      do
         ynormer = nn.Sequential()
         ynormer:add(nn.Narrow(1,1,1))
         ynormer:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(filterSize), normthres))
         normer = nn.ConcatTable()
         normer:add(ynormer)
         normer:add(nn.Narrow(1,2,2))
      end
      preproc:add(normer)
      preproc:add(nn.JoinTable(1))
   elseif opt.preproc == 'norm(y)' then
      planes = 1
      preproc:add(nn.SpatialColorTransform('rgb2y'))
      preproc:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(filterSize), normthres))
   elseif opt.preproc == 'rgb' then
      preproc:add(nn.Identity())
   elseif opt.preproc == 'yuv' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
   else
      print('incorrect arg: preproc')
      op:help()
      os.exit()
   end
end

print '==> generating CNN network:'
local CNN = nn.Sequential()

if opt.type == 'cuda' then
   -- CUDA needs the batch dimension to be the inner most (=> transposition)
   CNN:add(nn.Transpose({1,4},{1,3},{1,2}))

   -- stage 1: conv+max
   CNN:add(nn.SpatialConvolutionCUDA(ivch, nk1, is1, is1))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPoolingCUDA(ss1, ss1, ss1, ss1))

   -- stage 2: conv+max
   CNN:add(nn.SpatialConvolutionCUDA(nk1, nk2, is2, is2))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPoolingCUDA(ss2, ss2, ss2, ss2))

   -- stage 3: conv+max
   CNN:add(nn.SpatialConvolutionCUDA(nk2, nk3, is3, is3))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPoolingCUDA(ss3, ss3, ss3, ss3))

   -- CUDA reverse the transposition for integrability
   CNN:add(nn.Transpose({1,4},{2,4},{3,4}))
else
   -- stage 1: conv+max
   CNN:add(nn.SpatialConvolutionMM(ivch, nk1, is1, is1))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPooling(ss1, ss1, ss1, ss1))

   -- stage 2: conv+max
   CNN:add(nn.SpatialConvolutionMM(nk1, nk2, is2, is2))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPooling(ss2, ss2, ss2, ss2))

   -- stage 2: conv+max
   CNN:add(nn.SpatialConvolutionMM(nk2, nk3, is3, is3))
   CNN:add(nn.Threshold())
   CNN:add(nn.SpatialMaxPooling(ss3, ss3, ss3, ss3))
end

----------------------------------------------------------------------
-- Classifier (trainable with mini-batch)
-- a 2-layer perceptron
local classifier = nn.Sequential()
classifier:add(nn.Reshape(cl_nk1))
classifier:add(nn.Linear(cl_nk1,cl_nk2))
classifier:add(nn.Threshold())
classifier:add(dropout)
classifier:add(nn.Linear(cl_nk2,#classes))

-- final stage: log probabilities
classifier:add(nn.LogSoftMax())

-- putting network together: a <Sequential> of <Sequential>s
--    <model>
--       |___<CNN>
--       |___<classifier>

-- adjust all biases for threshold activation units
for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end
for _,layer in ipairs(classifier.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #classifier.modules-1 then
         layer.bias:zero()
      end
   end
end

model = nn.Sequential()
model:add(CNN)
model:add(classifier)

----------------------------------------------------------------------
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
   dropout = dropout
}

