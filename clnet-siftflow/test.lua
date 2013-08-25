----------------------------------------------------------------------
-- This script tests a trained network on the Siftflow dataset
--
-- Author: Clement Farabet
----------------------------------------------------------------------

require 'xlua'
require 'image'
require 'nnx'
require 'imgraph'
require 'optim'

----------------------------------------------------------------------
-- Parse options
--
dname,fname = sys.fpath()
op = xlua.OptionParser('%prog [options]')

op:option{'-n', '--network', action='store', dest='network',
          help='path to existing [trained] network', default='fovea.net'}

op:option{'-d', '--dataset', action='store', dest='dataset',
          help='path to dataset',
          default='../datasets/siftflow_dataset'}

op:option{'-dp', '--display', action='store_true', dest='display',
          help='show results live', default=false}

op:option{'-t', '--threshold', action='store', dest='threshold',
          help='segmentation threshold (btwn 0 and 10), 0 bypasses segmentation', 
          default=1}
op:option{'-c', '--confidence', action='store', dest='confidence',
          help='if segmentation used, set min confidence for hist pooling', 
          default=0}

op:option{'-sp', '--superpixels', action='store', dest='superpixels',
          help='use superpixels, must be available in dataset/subset-*/SUPERPIXELS', 
          default=false}

op:option{'-g', '--geometry', action='store', dest='geometry',
          help='resize test images to given geometry', 
          default='256'}

op:option{'-sv', '--save', action='store', dest='save',
          help='save to path'}

opt = op:parse()
op:summarize()

----------------------------------------------------------------------
-- Classes + Colors
--
classes = {'unknown',
           'awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus',
           'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field',
           'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river',
           'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky',
           'staircase', 'streetlight', 'sun', 'tree', 'window'}

colormap = imgraph.colormap{[1] ={0.0, 0.0, 0.0},
                            [2] ={0.5, 0.5, 0.5}, -- awning
                            [3] ={0.9, 0.3, 0.3}, -- balcony
                            [4] ={0.8, 0.3, 0.2}, -- bird
                            [5] ={0.4, 0.4, 0.8}, -- boat
                            [6] ={0.5, 0.9, 0.9}, -- bridge
                            [7] ={0.7, 0.7, 0.3}, -- building
                            [8] ={0.4, 0.7, 0.8}, -- bus
                            [9] ={0.4, 0.4, 0.8}, -- car
                            [10]={0.8, 0.6, 0.6}, -- cow
                            [11]={0.9, 0.7, 0.9}, -- crosswalk
                            [12]={0.9, 0.9, 0.5}, -- desert
                            [13]={0.5, 0.3, 0.0}, -- door
                            [14]={0.6, 0.5, 0.1}, -- fence
                            [15]={0.7, 0.7, 0.1}, -- field
                            [16]={0.0, 0.9, 0.0}, -- grass
                            [17]={0.0, 0.2, 0.2}, -- moon
                            [18]={0.7, 0.5, 0.3}, -- mountain
                            [19]={1.0, 0.0, 0.3}, -- person
                            [20]={0.3, 0.7, 0.1}, -- plant
                            [21]={0.4, 0.2, 0.2}, -- pole
                            [22]={0.1, 0.4, 0.9}, -- river
                            [23]={0.3, 0.3, 0.3}, -- road
                            [24]={0.5, 0.4, 0.2}, -- rock
                            [25]={0.8, 0.8, 0.5}, -- sand
                            [26]={0.1, 0.1, 0.9}, -- sea
                            [27]={0.5, 0.5, 0.5}, -- sidewalk
                            [28]={1.0, 0.1, 0.1}, -- sign
                            [29]={0.0, 0.7, 0.9}, -- sky
                            [30]={0.9, 0.4, 0.3}, -- staircase
                            [31]={0.1, 1.0, 0.1}, -- streetlight
                            [32]={1.0, 1.0, 0.0}, -- sun
                            [33]={0.2, 0.8, 0.1}, -- tree
                            [34]={0.1, 0.6, 1.0}} -- window

----------------------------------------------------------------------
-- Load network to test
--
print('<tester> reloading previously trained network')
network = torch.load(opt.network)
torch.setdefaulttensortype(torch.typename(network.output))

-- replace classifier (2nd module) by SpatialClassifier
convnet = network.modules[1]
classifier = network.modules[2]
network.modules[2] = nn.SpatialClassifier(classifier)

----------------------------------------------------------------------
-- Test network on dataset
--
confusion =  optim.ConfusionMatrix(classes)

-- gaussian (a gaussian, really, is always useful)
gaussian = image.gaussian(3)

-- input geometry
pcall(loadstring("width = " .. opt.geometry))

-- save ?
if opt.save then
   os.execute('mkdir -p ' .. opt.save)
   logfile = io.open(opt.save .. '/confusion.txt', 'w')
end

-- load test set
testData = nn.DataSetLabelMe{path=sys.concat(opt.dataset,'test'),
                          verbose=true,
                          nbClasses=#classes,
                          rawSampleMaxSize=width,
                          classNames=classes,
                          classToSkip=1}

for i = 1,#testData.rawdata do
   -- (0) confusion
   sconfusion = optim.ConfusionMatrix(classes)

   -- (1) load next sample to test
   testData:loadSample(i)
   img = testData.currentSample
   mask = testData.currentMask
   img = image.scale(img, width, width)
   mask = image.scale(mask, width, width, 'simple')

   -- (2) compute network's output
   distributions = network:forward(img)

   -- (3) upsample the distributions
   distributions = image.scale(distributions, img:size(3), img:size(2))

   -- (4) compute superpixels
   do
      -- (a) compute graph on input image
      smoothed = image.convolve(img, gaussian, 'same')
      graph = imgraph.graph(smoothed)

      -- (b) cut graph
      mstsegm = imgraph.segmentmst(graph, tonumber(opt.threshold), 30)

      -- (c) pool the predicted distributions into the segmentation
      distributions, icomponents = imgraph.histpooling(distributions, mstsegm, true, opt.confidence)
   end

   -- (5) winner take all
   _,winners = distributions:max(1)
   winner = winners[1]

   -- (6) compare with groundtruth
   for y = 1,winner:size(1) do
      for x = 1,winner:size(2) do
         if mask[y][x] ~= 1 then -- discard unknown class
            sconfusion:add(winner[y][x], mask[y][x])
            confusion:add(winner[y][x], mask[y][x])
         end
      end
   end

   -- (7) display
   if opt.display then
      -- () window
      require 'qt'
      require 'qtwidget'
      win = win or qtwidget.newwindow(width, width, 'Scene Segmenter')

      -- (a) infer nb of classes
      nclasses = distributions:size(1)

      -- (b) colorize classes
      colored, colormap = imgraph.colorize(winner, colormap)

      -- (c) overlay segmentation on input frame
      image.display{image=colored:add(img), win=win, min=0, max=2}
   end

   -- (8) save
   if opt.save then
      -- idx for saving
      idx = string.format('%05d',i)

      -- save results
      maskc, colormap = imgraph.colorize(mask, colormap)
      winnerc, colormap = imgraph.colorize(winner, colormap)
      image.save(opt.save .. '/label-' .. idx .. '.png', maskc:add(img):div(2))
      image.save(opt.save .. '/output-' .. idx .. '.png', winnerc:add(img):div(2))
      if opt.display then
         image.save(opt.save .. '/output+labels-' .. idx .. '.png', win:image():toTensor())
      end
      image.save(opt.save .. '/input-' .. idx .. '.png', img)

      -- save confusion
      logfile:write('img - ' .. idx .. '\n')
      logfile:write(tostring(sconfusion) .. '\n')
      logfile:flush()
   end

   -- print confusion
   print(sconfusion)
end

-- final confusion
print(sys.COLORS.red .. tostring(confusion) .. sys.COLORS.none)

-- export
if confusion.render then
   image.save('siftflow-confusion.png', confusion:render('score'))
end
os.exit()
