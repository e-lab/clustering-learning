----------------------------------------------------------------------
-- test videoknet.lua
-- Massive online trained network on videos
-- load all sort of video, run Clustering learning, online-learn forever
-- February 5th 2013, E. Culurciello

-- test if linear layer is any good on pixels:


print "==>loading and processing face dataset:"
dofile '1_data_faces.lua'


orig_sz = trainData[1][1]:size(2) -- size of image in dataset
--in_sz = 2*orig_sz -- 3 layers
in_sz = orig_sz

trainsize = trainData:size()
testsize = testData:size()
trsize = trainsize -- used by train function below

model = nn.Sequential()
model:add(nn.Reshape(in_sz^2))
model:add(nn.Linear(in_sz^2,2)) -- just two output: face, bg neuron

-- original face model:
--   model = nn.Sequential()
--   model:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(5)))
--   model:add(nn.SpatialConvolution(1, 8, 5, 5))
--   model:add(nn.Tanh())
--   model:add(nn.SpatialMaxPooling(4, 4, 4, 4))
--   model:add(nn.SpatialConvolutionMap(nn.tables.random(8, 64, 4), 7, 7))
--   model:add(nn.Tanh())
--   model:add(nn.Reshape(64))
--   model:add(nn.Linear(64,2))

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
         local sample = dataset[i]
         local input = sample[1]
         local target = sample[2]
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
      local sample = dataset[t]
      local input = sample[1]
      local target = sample[2]

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
while true do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   --trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   --testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --trainLogger:plot()
   --testLogger:plot()
end





