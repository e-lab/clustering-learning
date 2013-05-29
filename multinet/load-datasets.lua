------------------------------------------------------------------------------
-- Loading and concatenating all datasets
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'pl'

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> Processing options'
   opt = lapp [[
--German          Use the German road sign dataset
]]
end

opt = opt or {}

-- Parameters ----------------------------------------------------------------
torch.setnumthreads(3)

-- Main IF -------------------------------------------------------------------
if paths.filep('train.t7') and paths.filep('test.t7') then

   print '==> loading previously generated dataset:'
   trainData = torch.load('train.t7')
   testData = torch.load('test.t7')

else

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
   if opt.German then roadSign = 'data-sign' else roadSign = 'data-AmericanSign' end
   local datasets = {
      'data-person',
      roadSign,
      'data-kitti'
   }

   -- Number of classes per dataset
   if opt.German then nbSign = 43 else nbSign = 35 end
   local classes = {
      1,
      nbSign,
      1
   }

   -- Computing cumulative number of classes
   local totNbClasses = {0}
   for i,c in ipairs(classes) do
      totNbClasses[i+1] = totNbClasses[i] + c
   end

   -- Cleaning the screen before executing (a lot of text out will follow)
   -- os.execute('clear')

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

   -- Printing the category statistics ------------------------------------------
   print '==> showing class statistics'
   gnuplot.figure(1)
   gnuplot.hist(trainData.labels,totNbClasses[#totNbClasses])
   gnuplot.title('Training dataset category statistics')
   gnuplot.figure(2)
   gnuplot.hist(testData.labels,totNbClasses[#totNbClasses])
   gnuplot.title('Testing dataset category statistics')

   -- Save created dataset ------------------------------------------------------
   print '==> saving dataset on disk'
   torch.save('train.t7',trainData)
   torch.save('test.t7',testData)

end

-- Done :) -------------------------------------------------------------------
print '==> Datasets loaded successfully! :)'
