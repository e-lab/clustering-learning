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
require 'xlua'

-- Exporting functions to the global namespace -------------------------------
ls = eex.ls

-- Parsing the command line --------------------------------------------------
if not opt then
   print '==> Processing options'
   opt = lapp [[
   --playDataset visualise the whole signs dataset
   --firstFrame (default 21)
   --lastFrame  (default 30)
   -h, --height (default 46)
   -w, --width (default 46)
]]
end

-- Parameters ----------------------------------------------------------------
ds = eex.datasetsPath()
path = ds .. 'GTSRB/Final_Training/Images/'
height = opt.height
width = opt.width

-- Main program -------------------------------------------------------------
print '==> creating a new dataset from raw files:'
totNbSign = 0
for i = 1, #ls(path) do
   nbSign = #ls(path..ls(path)[i]..'/*.png')/30
   totNbSign = totNbSign + nbSign
end
dsSize = totNbSign * (opt.lastFrame - opt.firstFrame + 1)

dataset = {
   data = torch.Tensor(dsSize,3,height,width),
   labels = torch.Tensor(dsSize),
   size = function() return dsSize end
}

-- Load, crop and resize image
idx = 0
for i = 1, #ls(path) do -- loop over different signs type
   nbSign = #ls(path..ls(path)[i]..'/*.png')/30
   for j = 1, nbSign do -- loop over different sample of same sign type
      for k = opt.firstFrame, opt.lastFrame do -- loop over different frames of the same physical sign
         img = image.load(string.format('%s%s/%05d_%05d.png',path,ls(path)[i],j-1,k-1))
         w,h = (#img)[3],(#img)[2]
         min = (w < h) and w or h
         idx = idx + 1
         img  = image.crop(img,math.floor((w-min)/2),math.floor((h-min)/2),w-math.ceil((w-min)/2),h-math.ceil((h-min)/2))
         image.scale(img,dataset.data[idx])
         xlua.progress(idx,dsSize)
      end
   end
end


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
