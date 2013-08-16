----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet & Alfredo Canziani (May 2013)
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss
local dropout = t.dropout

-- This matrix records the current testConfusion across classes (<classes> is a global var.)
local testConfusion = optim.ConfusionMatrix(classes)

-- Storing max test accuracy (for conditional network saving)
local maxTestAccSoFar = 0

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
   dropout.train = false

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         if opt.siftflow then
            local a = testData[idx]
            inputs[idx] = preproc:forward(a[1])
            targets[idx] = a[2]:gt(0):float() * l
         else
            inputs[idx] = testData.data[i]
            targets[idx] = testData.labels[i]
         end
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward(inputs)

      -- testConfusion
      for i = 1,opt.batchSize do
         testConfusion:add(preds[i], targets[i])
      end
   end
   xlua.progress(testData:size(), testData:size())

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print testConfusion matrix
   print(testConfusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = testConfusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- save current net IF test accuracy has increased
   if testConfusion.totalValid > maxTestAccSoFar then
      print('==> test accuracy has increased to ' .. testConfusion.totalValid*100 .. '%, (+' .. (testConfusion.totalValid-maxTestAccSoFar)*100 .. '%) => saving network')
      maxTestAccSoFar = testConfusion.totalValid
      saveNet('multinet.net')
   end

   testConfusion:zero()

   -- dropout -> on
   dropout.train = true
end

-- Export:
return test
