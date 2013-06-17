----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'Dropout'

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end


----------------------------------------------------------------------
-- define network to train

print '==> Defining network parameters:'

local nnf1,nnf2,nnf3  = 1,1,1         -- number of frames at each layer
local nk0,nk1,nk2,nk3 = 3,16,32,64    -- nb of features
local is0,is1,is2,is3 = 15,7,5,5      -- size of kernels
local ss1,ss2,ss3     = 2,2,4         -- size of subsamplers (strides)
local hiddens         = 64            -- nb of hidden features for top perceptron (0=linear classifier)
local cl_nk1,cl_nk2   = nk3, hiddens  -- dimensions for top perceptron
local ivch            = 3

-- Old net
--[[nnf1,nnf2,nnf3  = 1,1,1             -- number of frames at each layer
nk0,nk1,nk2,nk3 = 3,16,128,256      -- nb of features
is0,is1,is2,is3 = 15,7,7,7          -- size of kernels
ss1,ss2         = 2,2               -- size of subsamplers (strides)
hiddens         = 512               -- nb of hidden features for top perceptron (0=linear classifier)
cl_nk1,cl_nk2   = nk3, hiddens      -- dimensions for top perceptron
ivch            = 3]]

-- dropout?
local dropout = nn.Dropout(opt.dropout)

----------------------------------------------------------------------

print '==> generating CNN network:'

local CNN = nn.Sequential()
CNN:add(nn.SpatialConvolutionMM(ivch, nk1, is1, is1))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
-- 2nd layer 
CNN:add(nn.SpatialConvolutionMM(nk1,nk2, is2, is2))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss2,ss2,ss2,ss2))
-- 3rd layer
CNN:add(nn.SpatialConvolutionMM(nk2,nk3, is3, is3))
CNN:add(nn.Threshold())
CNN:add(nn.SpatialMaxPooling(ss3,ss3,ss3,ss3))

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

local model = nn.Sequential()
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

