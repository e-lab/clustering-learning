----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

-- model:
--local t = require 'model'
--local model = t.model
--local loss = t.loss
--local dropout = t.dropout

-- classes
local classes = {'person','bg'}

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Batch test:
local inputs = torch.Tensor(opt.batchSize,testData.data:size(2),testData.data:size(3),testData.data:size(4))
local targets = torch.Tensor(opt.batchSize)
if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

   -- dropout -> off
   --dropout.p = 0

   -- test over test data
   print('==> testing on test set:')
   for t = 1,tesize,opt.batchSize do
      -- disp progress
      xlua.progress(t, tesize)

      -- batch fits?
      if (t + opt.batchSize - 1) > tesize then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = testData.data[i]
         targets[idx] = testData.labels[i]
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward(inputs)

      -- confusion
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   
   confusion:zero()
   
   -- dropout -> on
   --dropout.p = opt.dropout
end

-- Export:
return test

