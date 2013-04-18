----------------------------------------------------------------------
-- This script demonstrates how to load the Faces 
-- training data, and pre-process it to facilitate learning.
--
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, modified by E. Culurciello - jan 2013
----------------------------------------------------------------------

require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator
require 'math'

----------------------------------------------------------------------
-- create dataset
--
if not sys.dirp(opt.dataset) then
   local path = sys.dirname(opt.dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p ' .. path .. '; '..
              'cd ' .. path .. '; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
end

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

-- Faces:
dataFace = nn.DataSet{dataSetFolder=opt.dataset..'face', 
                      cacheFile=opt.dataset..'face',
                      nbSamplesRequired=opt.patches,
                      channels=1}
dataFace:shuffle()

-- Backgrounds:
dataBG = nn.DataSet{dataSetFolder=opt.dataset..'bg',
                    cacheFile=opt.dataset..'bg',
                    nbSamplesRequired=opt.patches,
                    channels=1}
dataBGext = nn.DataSet{dataSetFolder=opt.dataset..'bg-false-pos-interior-scene',
                       cacheFile=opt.dataset..'bg-false-pos-interior-scene',
                       nbSamplesRequired=opt.patches,
                       channels=1}
dataBG:appendDataSet(dataBGext)
dataBG:shuffle()

-- pop subset for testing
testFace = dataFace:popSubset{ratio=opt.ratio}
testBG = dataBG:popSubset{ratio=opt.ratio}

-- training set
trainData = nn.DataList()
trainData:appendDataSet(dataFace,'Faces')
trainData:appendDataSet(dataBG,'Background')

-- testing set
testData = nn.DataList()
testData:appendDataSet(testFace,'Faces')
testData:appendDataSet(testBG,'Background')

-- display
if opt.visualize then
   trainData:display(100,'trainData')
   testData:display(100,'testData')
end


-- clean up NaN and -inf in data (has to run multiple times):
--for j=1,3 do
--counter = 0
--for i = 1, trainData:size() do
--   if (torch.sum(trainData[i][1]-trainData[i][1]) ~= 0) or (torch.sum(trainData[i][1]) == -math.huge) then 
--      trainData[i][1] = torch.Tensor(1,32,32)
--      trainData[2][2][1] = -1 -- set to bg
--      trainData[2][2][2] = 1
--      counter = counter + 1
--   end
--end
--print('Number of NaN/-inf trainData', counter)
--for i = 1, testData:size() do
--   if (torch.sum(testData[i][1]-testData[i][1]) ~= 0) or (torch.sum(trainData[i][1]) == -math.huge) then 
--      testData[i][1] = torch.Tensor(1,32,32)
--      testData[2][2][1] = -1 -- set to bg
--      testData[2][2][2] = 1
--      counter = counter + 1
--    end
--end
--print('Number of NaN/-inf testData', counter)
--end


