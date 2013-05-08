------------------------------------------------------------------------------
-- Loading street signals
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013
------------------------------------------------------------------------------

-- Requires ------------------------------------------------------------------
require 'image'
require 'sys'

-- Useful functions ----------------------------------------------------------
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

-- Parameters ----------------------------------------------------------------
path = '../../datasets/GTSRB/Final_Training/Images/'
--sign = sign or 2

-- Main program -------------------------------------------------------------
print 'Visualising the dataset'
for sign = 1, #ls(path) do

   -- Showing a couple of images
   for i = 1,#ls(path .. ls(path)[sign] .. '/*.png'),1 do
      img = image.load(ls(path .. ls(path)[sign] .. '/*.png')[i])
      win = image.display{image=img,zoom=10,win=win}
      --io.read()
   end

end
