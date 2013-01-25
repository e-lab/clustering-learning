
-- 1st layer:
----------------------------------------------------------------------
print "==> processing dataset with CL network"

testData2 = {
   data = torch.Tensor(testsize, nk, l1netoutsize, l1netoutsize),
   size = function() return testsize end
}

trainData2 = {
   data = torch.Tensor(trainsize, nk, l1netoutsize, l1netoutsize),
   size = function() return trainsize end
}
for t = 1,trainsize do
   trainData2.data[t] = l1net:forward(trainData[t][1])
   xlua.progress(t, trainsize)
end

for t = 1,testsize do
   testData2.data[t] = l1net:forward(testData[t][1])
   xlua.progress(t, testsize)
end


-- relocate pointers to new dataset:
--trainData1 = trainData -- save original dataset
--testData1 = testData
--trainData = trainData2 -- relocate new dataset
--testData = testData2

-- show a few outputs:
if opt.visualize then
   f256S_y = trainData2.data[{ {1,256}}]:reshape(256,nk1,l1netoutsize,l1netoutsize)[{{},1,{},{}}]
   image.display{image=f256S_y, nrow=16, nrow=16, padding=2, zoom=2, 
            legend='Output 1st layer: first 256 examples, 1st feature'}
end

--print '==> verify statistics'
--channels = {'r','g','b'}
--for i,channel in ipairs(channels) do
--   trainMean = trainData.data[{ {},i }]:mean()
--   trainStd = trainData.data[{ {},i }]:std()
--
--   testMean = testData.data[{ {},i }]:mean()
--   testStd = testData.data[{ {},i }]:std()
--
--   print('training data, '..channel..'-channel, mean: ' .. trainMean)
--   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
--
--   print('test data, '..channel..'-channel, mean: ' .. testMean)
--   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
--end


----------------------------------------------------------------------
-- save datasets:
--if opt.savedataset then
--   trainData.data = trainData.data:float()  -- float to save space if needed
--   testData.data = testData.data:float()
--   torch.save('trainData-cifar-CL1l-dist.t7', trainData)
--   torch.save('testData-cifar-CL1l-dist.t7', testData)
--end

