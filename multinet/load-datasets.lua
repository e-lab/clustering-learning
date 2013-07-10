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
      --German                                  Use the German road sign dataset
      --lim                (default 50)         at least <lim> examples per sign, max 1000
      --maxBg              (default 1e9)        max number of background samples
      --samplepercar       (default 6)          number of the patch to extract from per car (bounding box)
      --dataStop                                use German stop signs
      --kittiRatio         (default 0.8)        train/tot ratio for kitti dataset
]]
end

opt = opt or {}

-- Parameters ----------------------------------------------------------------
torch.setnumthreads(3)
torch.setdefaulttensortype('torch.FloatTensor')

-- Main IF -------------------------------------------------------------------
if paths.filep('train.t7') and paths.filep('test.t7') then

   print '==> loading previously generated dataset:'
   trainData = torch.load('train.t7')
   testData = torch.load('test.t7')
   nbClasses = torch.load('nbClasses.t7')

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
   if opt.German then
      roadSign = 'data-sign'
   elseif opt.dataStop then
      roadSign = 'data-stop'
   else roadSign = 'data-AmericanSign' end
   local datasets = {
      'data-person',
      roadSign,
      'data-kitti2'
   }

   nbClasses = {}
   local totNbClasses = {0}

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

      nbClasses[i] = torch.max(data.trainData.labels) + 1
      totNbClasses[i+1] = totNbClasses[i] + nbClasses[i]

   end

   -- Exporting main dataset to the global namespace
   trainData = totalTrainData
   testData  = totalTestData

   -- Printing the category statistics ------------------------------------------
   print '==> showing class statistics'
   gnuplot.figure(1)
   gnuplot.hist(trainData.labels,totNbClasses[#totNbClasses])
   gnuplot.title('Training dataset category statistics')
   gnuplot.axis({'','',0,''})
   gnuplot.figure(2)
   gnuplot.hist(testData.labels,totNbClasses[#totNbClasses])
   gnuplot.title('Testing dataset category statistics')
   gnuplot.axis({'','',0,''})

   -- Save created dataset ------------------------------------------------------
   print '==> saving dataset on disk'
   torch.save('train.t7',trainData)
   torch.save('test.t7',testData)
   torch.save('nbClasses.t7',nbClasses)

end

-- Done :) -------------------------------------------------------------------
print '==> Datasets loaded successfully! :)'
