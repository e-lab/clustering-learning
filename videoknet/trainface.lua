-- train face

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
-- training criterion: a simple Mean-Square Error
--
criterion = nn.MSECriterion()
criterion.sizeAverage = true
----------------------------------------------------------------------
-- train/test
--

dname = sys.fpath()
opt.save=dname .. '/results' .. '/face.net'

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix{'Face','Background'}

-- log results to files
trainLogger = optim.Logger(paths.concat(sys.dirname(opt.save), 'facetrain.log'))
testLogger = optim.Logger(paths.concat(sys.dirname(opt.save), 'facetest.log'))

-- optim config
config = {learningRate = 1e-3, weightDecay = 1e-3,
          momentum = 0.1, learningRateDecay = 5e-7}

batchSize = 1
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,dataset:size(),batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,dataset:size()) do
         -- load new sample
         --local sample = dataset[i]
         local input = dataset.data[i]--sample[1]
         local target = dataset.labels[i]--sample[2]
         table.insert(inputs, input)
         table.insert(targets, target)
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
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])

                          -- visualize?
--                          if opt.visualize then
--                             display(inputs[i])
--                          end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optim.sgd(feval, parameters, config)
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
--   local filename = opt.save
--   os.execute('mkdir -p ' .. sys.dirname(filename))
--   if sys.filep(filename) then
--      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
--   end
--   print('<trainer> saving network to '..filename)
   --torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      --local sample = dataset[t]
      local input = dataset.data[t]--sample[1]
      local target = dataset.labels[t]--sample[2]

      -- test sample
      confusion:add(model:forward(input), target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

----------------------------------------------------------------------
-- and train!
--
for i = 1, 32 do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   --trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   --testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --trainLogger:plot()
   --testLogger:plot()
end





