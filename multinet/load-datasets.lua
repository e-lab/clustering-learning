------------------------------------------------------------------------------
-- Loading and concatenating all datasets
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Parameters ----------------------------------------------------------------
torch.setnumthreads(3)

-- Global functions ----------------------------------------------------------
function cat(a,b)
   if a and b then
      return torch.cat(a,b,1)
   else
      return a or b
   end
end

-- Main program -------------------------------------------------------------
totalTrainData = {}
totalTestData = {}

datasets = {
   'data-person',
   'data-sign',
   'data-kitti'
}

classes = {
   1,
   43,
   1
}

totNbClasses = {0}
for i,c in ipairs(classes) do
   totNbClasses[i+1] = totNbClasses[i] + c
end

os.execute('clear')

for i,d in ipairs(datasets) do
   data = require(d)

   totalTrainData.data = cat(totalTrainData.data,data.trainData.data)
   totalTrainData.labels = cat(totalTrainData.labels,data.trainData.labels + 1 + totNbClasses[i])
   totalTrainData.size = function() return (#totalTrainData.data)[1] end

   totalTestData.data = cat(totalTestData.data,data.testData.data)
   totalTestData.labels = cat(totalTestData.labels,data.testData.labels + 1 + totNbClasses[i])
   totalTestData.size = function() return (#totalTestData.data)[1] end

end

trainData = totalTrainData
testData  = totalTestData
