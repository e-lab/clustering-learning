------------------------

function epoch()
   -- train for one epoch on current subset
   print('<trainer> on training set:')
   time = sys.clock()
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainsize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      local options = {}
      for i = t,math.min(t+batchSize-1,trainsize) do
         -- load new sample
         local sample = trainData2[i] -- pick samples processed by unsup network
         local input = sample[1]
         local target = sample[2]
--         local sample_x = sample[3]
--         local sample_y = sample[4]
--         local sample_size = sample[5]

         -- store input/target
         table.insert(inputs, input)
         table.insert(targets, target)
         table.insert(options, {x=sample_x, y=sample_y, size=sample_size})
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- focus fovea
                          --fovea:focus(options[i].x, options[i].y, options[i].size)

                          -- estimate f
                          local output = trainable:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          trainable:backward(inputs[i], df_do)

                          -- update confusion matrix
                          if trainConfusion then
                             trainConfusion:add(output, targets[i])
                          end

                          -- visualize?
                          if opt.visualize then
                             display(inputs[i])
                          end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize
      --if inputs[1]:size(1) == 3 then
         -- optimize the model given current input/target set
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = opt.learningRateDecay}
         _,fx = optim.sgd(feval, parameters, config)
      --else
      --   print('<trainer> warning: skipping sample with only ' .. inputs[1]:size(1) .. ' channel(s)')
      --end
   end

   -- average processing time
   time = sys.clock() - time
   time = time / trainsize
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- train error/confusion
   print(trainConfusion)

   -- free up memory
   fovea.cachedPreProcessed = nil
   collectgarbage()

   -- create test net
   testable = trainable:clone()
   --testable.modules[1]:focus() -- defocus
   testable = nn.SpatialClassifier(testable) -- spatial classifier

   -- then test
   time = sys.clock()
   ntests = 0
   for i = 1,#testData.rawdata/5 do
      -- disp progress
      xlua.progress(i, #testData.rawdata/5)

      -- load new sample
      --testData:loadSample((i-1)*5+1)
      local input = testData2[i][1] -- test only one processed by unsup network
      local mask = testData2[i][2]

      -- test sample
      local output = testable:forward(input)
      mask = image.scale(mask, output:size(3), output:size(2))

      -- loop over all locations
      local target = torch.Tensor(#classes):zero()
      for y = 1,(#mask)[1] do
         for x = 1,(#mask)[2] do
            -- target at x,y location
            target:zero()
            target[mask[{ y,x }]] = 1

            -- update confusion matrix / error
            if mask[{ y,x }] ~= 1 then
               testConfusion:add(output[{ {},y,x }], target)
               ntests = ntests + 1
            end
         end
      end
   end

   -- average processing time
   time = sys.clock() - time
   time = time / ntests
   print("<tester> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- train error/confusion
   print(testConfusion)

   -- save network if error if test error is better
   averageValid = averageValid or 0
   if opt.save and (testConfusion.averageValid > averageValid) then
      print('<trainer> best average accuracy reached: ' .. (testConfusion.averageValid*100)
         .. '%, saving network to '..opt.save)
      fovea:focus()
      torch.save(opt.save, trainable)
      averageValid = testConfusion.averageValid
   end

   -- report on error/confusion
   logger:add {['Average Per-Class Accuracy [%] (train set)'] = trainConfusion.averageValid * 100,
               ['Pixelwise Accuracy [%] (train set)'] = trainConfusion.totalValid * 100,
               ['Average Per-Class Accuracy [%] (test set)'] = testConfusion.averageValid * 100,
               ['Pixelwise Accuracy [%] (test set)'] = testConfusion.totalValid * 100}
   if opt.plot then
      logger:style {['Average Per-Class Accuracy [%] (train set)'] = {'+','~ 1e-3'},
                    ['Pixelwise Accuracy [%] (train set)'] = {'+','~ 1e-3'},
                    ['Average Per-Class Accuracy [%] (test set)'] = {'+','~ 1e-3'},
                    ['Pixelwise Accuracy [%] (test set)'] = {'+','~ 1e-3'}}
      logger:plot()
   end

   -- reset matrices
   trainConfusion:zero()
   testConfusion:zero()

   -- free up memory
   fovea.cachedPreProcessed = nil
   collectgarbage()
end

-- train !
while true do 
   epoch() 
end
