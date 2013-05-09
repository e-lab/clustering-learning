------------------------------------------------------------------------------
-- Loading street signs
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'
require 'sys'
require 'pl'
require 'eex'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> Processing options'
   opt = lapp [[
   --playDataset visualise the whole signs dataset
]]
end

-- Parameters ----------------------------------------------------------------
ds = eex.datasetsPath()
path = ds .. 'GTSRB/Final_Training/Images/'
print(path)

-- Main program -------------------------------------------------------------
-- Play the dataset, if requested
if opt.playDataset then
   print 'Visualising the dataset'
   for sign = 1, #ls(path) do

      -- Showing a couple of images
      for i = 1,#ls(path .. ls(path)[sign] .. '/*.png'),1 do
         img = image.load(ls(path .. ls(path)[sign] .. '/*.png')[i])
         win = image.display{image=img,zoom=10,win=win}
         --io.read()
      end

   end
end
