----------------------------------------------------------------------
-- test videoknet.lua
-- Massive online trained network on videos
-- load all sort of video, run Clustering learning, online-learn forever
-- February 5th 2013, E. Culurciello

-- test on videos of faces also

function processVnet(network, data_in, nfpr, nkernels, oheight, owidth) -- pass frames into full trained vnet
   data_out = torch.Tensor(nfpr, nkernels, oheight, owidth)
   for i = nnf1, nfpr do -- just get a few frames to begin with
      procFrames = data_in[{{i-nnf1+1,i},{},{}}]:transpose(1,2) -- swap order of indices here for VolConvolution to work
      data_out[i] = network:forward(procFrames)
      xlua.progress(i, nfpr)
      -- do a live display of the input video and output feature maps 
      winm = image.display{image=data_out[i], padding=2, zoom=1, win=winm}
   end
   -- data_out = nil --free memory if needed
   return data_out
end

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
--vnet = torch.load('tnet.net')

print "==>loading and processing video dataset:"

-- load bg set:
dspath = '/Users/eugenioculurciello/Desktop/bagr.mov'
source_bg = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=false, load=false}

-- load face set: 
dspath = '/Users/eugenioculurciello/Desktop/face.mov'
source_fc = ffmpeg.Video{path=dspath, encoding='jpg', fps=24, loaddump=false, load=false}


trainsize = 200
testsize = 100

-- repeated from main train-videoknet, maybe adapt there!
function createDataBatch(trainsize,trainData,source)
   trainData = torch.Tensor(trainsize,ivch,ivhe,ivwi)
   for i = 1, nfpr do -- just get a few frames to begin with
      procFrame = normalization:forward(rawFrame) -- full LCN!
      trainData[i] = procFrame
      rawFrame = source:forward()
   end
   return trainData
end

-- load and normalize frames of both sets:
databg = createDataBatch(trainsize+testsize,trainData,source_bg)
datafc = createDataBatch(trainsize+testsize,trainData,source_fc)

answer = tnet:forward(datafc[{{nnf1,1},{},{}}]:transpose(1,2)):size()
nko  = answer[1]
ovhe = answer[2]
ovwi = answer[3]

print "==> processing  dataset frames with videoknet:"
processedbg = processVnet(tnet, databg, (trainsize+testsize)/2, nko, ovhe, ovwi) 
processedfc = processVnet(tnet, datafc, (trainsize+testsize)/2, nko, ovhe, ovwi)


print "==> creating split/randomize/alternated video dataset:"
testData2 = {
   data = torch.Tensor(testsize, nko, ovhe, ovwi),
   labels = torch.Tensor(testsize,2),
   size = function() return testsize end
}

trainData2 = {
   data = torch.Tensor(trainsize, nko, ovhe, ovwi),
   labels = torch.Tensor(trainsize,2),
   size = function() return trainsize end
}

for t = 1,trainsize,2 do -- alternate fc and bg:
   trainData2.data[t] =  processedbg[(t+1)/2] -- bg
   trainData2.labels[t] = torch.Tensor({-1,1}) -- bg
   trainData2.data[t+1] = processedfc[(t+1)/2] -- fc
   trainData2.labels[t+1] = torch.Tensor({1,-1}) -- fc
   xlua.progress(t, trainsize)
end
for t = 1,testsize,2 do -- alternate fc and bg:
   testData2.data[t] = processedbg[(trainsize+t+1)/2] -- bg
   testData2.labels[t] = torch.Tensor({-1,1}) -- bg
   testData2.data[t+1] = processedfc[(trainsize+t+1)/2] -- fc
   testData2.labels[t+1] = torch.Tensor({1,-1}) -- fc
   xlua.progress(t, testsize)
end

print "==> train/test:"

-- relocate pointers to new dataset:
--trainData1 = trainData -- save original dataset
--testData1 = testData
trainData = trainData2 -- relocate new dataset
testData = testData2


trsize = trainsize -- used by train function below

model = nn.Sequential()
model:add(nn.Reshape(nko*ovhe*ovwi))
model:add(nn.Linear(nko*ovhe*ovwi,2)) -- just two output: face, bg neuron


dofile 'trainface.lua'

