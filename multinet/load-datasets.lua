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
-- Temporary dataset container
local totalTrainData = {}
local totalTestData = {}

-- Datasets' name
local datasets = {
   'data-person',
   'data-sign',
   'data-kitti'
}

-- Number of classes per dataset
local classes = {
   1,
   43,
   1
}

-- Computing cumulative number of classes
local totNbClasses = {0}
for i,c in ipairs(classes) do
   totNbClasses[i+1] = totNbClasses[i] + c
end

-- Cleaning the screen before executing (a lot of text out will follow)
os.execute('clear')

-- Concatenating all datasets contained in <dataset> table
for i,d in ipairs(datasets) do

   local data = require(d)

   totalTrainData.data = cat(totalTrainData.data,data.trainData.data)
   totalTrainData.labels = cat(totalTrainData.labels,data.trainData.labels + 1 + totNbClasses[i])
   totalTrainData.size = function() return (#totalTrainData.data)[1] end

   totalTestData.data = cat(totalTestData.data,data.testData.data)
   totalTestData.labels = cat(totalTestData.labels,data.testData.labels + 1 + totNbClasses[i])
   totalTestData.size = function() return (#totalTestData.data)[1] end

end

-- Exporting main dataset to the global namespace
trainData = totalTrainData
testData  = totalTestData

-- Done :)
print '==> Datasets loaded successfully! :)'
