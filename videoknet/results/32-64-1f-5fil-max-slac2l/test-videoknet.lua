----------------------------------------------------------------------
-- test videoknet.lua
-- Massive online trained network on videos
-- load all sort of video, run Clustering learning, online-learn forever
-- February 5th 2013, E. Culurciello


print "==>loading and processing face dataset:"
dofile '1_data_faces.lua'

-- prepare full network with all layers:
tnet = vnet:clone()
for i=1,vnet2:size() do
   tnet:add(vnet2.modules[i]:clone())
end
nk = nk2
--nk = nk3 -- 3 layers
--for i=1,vnet3:size() do
--   tnet:add(vnet3.modules[i]:clone())
--end
--torch.save('tnet.net',tnet)

orig_sz = trainData[1][1]:size(2) -- size of image in dataset
--in_sz = 2*orig_sz -- 3 layers
in_sz = orig_sz
tnet:forward(torch.Tensor(ivch,nnf1,72,72)) -- test network
temp = image.scale(trainData[1][1], in_sz, in_sz)
vkn_osz = tnet:forward(temp:resize(1,1,in_sz,in_sz):expand(3,1,in_sz,in_sz):expand(3,nnf1,in_sz,in_sz)):size(2) -- output size of network

trainsize = trainData:size()
testsize = testData:size()


print "==> processing dataset with videoknet:"
testData2 = {
   data = torch.Tensor(testsize, nk, vkn_osz, vkn_osz),
   labels = torch.Tensor(testsize,2),
   size = function() return testsize end
}

trainData2 = {
   data = torch.Tensor(trainsize, nk, vkn_osz, vkn_osz),
   labels = torch.Tensor(trainsize,2),
   size = function() return trainsize end
}

for t = 1,trainsize do
   temp = trainData[t][1]
   --temp = image.scale(trainData[t][1], in_sz, in_sz) -- 3 layer, process on 2x scaled dataset
   trainData2.data[t] = tnet:forward(temp:resize(1,1,in_sz,in_sz):expand(3,1,in_sz,in_sz):expand(3,nnf1,in_sz,in_sz))
   trainData2.labels[t] = trainData[t][2]
   xlua.progress(t, trainsize)
end
for t = 1,testsize do
   temp = testData[t][1]
   --temp = image.scale(testData[t][1], in_sz, in_sz) -- 3 layer ,process on 2x scaled dataset
   testData2.data[t] = tnet:forward(temp:resize(1,1,in_sz,in_sz):expand(3,1,in_sz,in_sz):expand(3,nnf1,in_sz,in_sz))
   testData2.labels[t] = testData[t][2]
   xlua.progress(t, testsize)
end


-----------------------------------------------------------------
-- Find face neuron at output of network:
--dofile 'faceneurontest.lua'

--torch.load() -- break function
-----------------------------------------------------------------


print "==> classic SGD train/test:"

-- relocate pointers to new dataset:
--trainData1 = trainData -- save original dataset
--testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2


trsize = trainsize -- used by train function below

model = nn.Sequential()
model:add(nn.Reshape(nk*vkn_osz^2))
model:add(nn.Linear(nk*vkn_osz^2,2)) -- just two output: face, bg neuron


dofile 'trainface.lua'

