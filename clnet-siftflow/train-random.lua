
----------------------------------------------------------------------
-- test with ininitialized random net: sanity check!


----------------------------------------------------------------------
-- A simple script that trains a multiscale network (the so-called
-- spatial fovea) on the the Siftflow Dataset.
--
-- Author: Clement Farabet
-- modified by Eugenio Culurciello, Feb 2013 for Clustering Learning
-- we train unsup on videos and test on a dataset - here SIFT flow
----------------------------------------------------------------------
-- TODO: update nn.fovea to allow for nn.VolumetricConvolution
----------------------------------------------------------------------

--require 'xlua'
require 'image'
require 'nnx'
require 'optim'
-- ec added:
require 'eex'
require 'online-kmeans' -- allow you to re-train k-means kernels
require 'ffmpeg'
require 'trainLayer' -- functions for Clustering Learning on video
require "slac"

----------------------------------------------------------------------
-- parse options
--
dname,fname = sys.fpath()
op = xlua.OptionParser('%prog [options]')

op:option{'-n', '--network', action='store', dest='network',
          help='path to existing [trained] network'}
op:option{'-s', '--save', action='store', dest='save',
          help='file name to save network [saving is done after each epoch]',
          default='scratch/' .. fname:gsub('.lua','') .. '_PARAMS/fovea.net'}

op:option{'-d', '--dataset', action='store', dest='dataset',
          help='path to dataset',
          default='../datasets/siftflow_dataset'}
op:option{'-w', '--www', action='store', dest='www', 
          default='http://data.neuflow.org/data/siftflow_dataset.tgz',
          help='path to retrieve dataset online (if not available locally)'}
op:option{'-c', '--clear', action='store_true', dest='clearcache', default=false,
          help='clear dataset cache to force reconstruction'}
op:option{'-j', '--jitter', action='store_true', dest='jitter', default=false,
          help='add jitter to dataset: rotations and horizontal flip'}
op:option{'-sm', '--sampling', action='store', dest='sampling', default='equal',
          help='sampling mode: random (frequentist training, favors large objects) OR equal (favors no one)'}
op:option{'-dt', '--distort', action='store_true', dest='distort', default=false,
          help='distort images and targets at runtime'}

op:option{'-dp', '--display', action='store_true', dest='display', default=false,
          help='display training/testing samples while training'}
op:option{'-plt', '--plot', action='store_true', dest='plot', default=false,
          help='plot error/accuracy live (if false, still logged in a file)'}
op:option{'-log', '--log', action='store_true', dest='log', default=false,
          help='log the whole session to a file'}

op:option{'-p', '--preproc', action='store', dest='preproc', default='norm(y)+norm(u)+norm(v)',
          help='preprocessing type, can be one of : norm(rgb) OR norm(yuv) OR norm(y)+norm(u)+norm(v) OR norm(y)+uv OR norm(y) OR rgb OR yuv'}

op:option{'-dst', '--distributions', action='store_true', dest='distributions',
          help='targets are distributions of classes, rather than 1-of-N vectors',
          default=false}

op:option{'-bs', '--batchSize', action='store', dest='batchSize',
          default=1,
          help='mini-batch size'}
op:option{'-mi', '--maxIteration', action='store', dest='bfgsMaxIteration',
          default=20,
          help='maximum nb of iterations for each mini-batch'}
op:option{'-lr', '--lrate', action='store', dest='learningRate', default=1e-3,
          help='learning rate'}
op:option{'-lrd', '--lratedecay', action='store', dest='learningRateDecay', default=1e-7,
          help='learning rate decay'}
op:option{'-wd', '--wdecay', action='store', dest='weightDecay', default=0,
          help='weight decay'}
op:option{'-wm', '--wmomentum', action='store', dest='momentum', default=0,
          help='weight momentum'}

op:option{'-sc', '--scales', action='store', dest='scales', default="1,2,4",
          help='scales'}
op:option{'-sh', '--share', action='store', dest='share',
          help='share weights across scales: "all" means all, 0 means none, '
             ..'1 means first layer, 2 first two layers...',
          default=0}
op:option{'-nf', '--nfeatures', action='store', dest='nfeatures', default="3,16,64,256",
          help='number of features, for each layer'}
op:option{'-nc', '--nconnex', action='store', dest='nconnex', default="0,1,8,32",
          help='number of connections, at each layer'}
op:option{'-ks', '--kernelsize', action='store', dest='kersize', default="15,7,7,7",
          help='kernel size, at each layer'}
op:option{'-ss', '--subsize', action='store', dest='subsize', default="2,2",
          help='subsampling size, at each layer'}
op:option{'-pl', '--pooling', action='store', dest='pooling', default="max",
          help='subsampling/pooling type: max OR sum OR l2'}
op:option{'-hid', '--hiddens', action='store', dest='hiddens', default="1024",
          help='nb of hidden features for top perceptron (if 0, a simple linear classifier is used)'}
op:option{'-rbf', '--rbf', action='store_true', dest='rbf', default=false,
          help='use rbf output units, instead of regular linear units'}

op:option{'-t', '--type', action='store', dest='type', default='double',
          help='numeric type: float | double'}
op:option{'-sd', '--seed', action='store', dest='seed', default=1,
          help='use fixed seed for randomized initialization'}
op:option{'-thd', '--threads', action='store', dest='threads', default=8,
          help='use multiple threads for function eval'}

opt = op:parse()
op:summarize()
parsed = op:tostring(true, {'-nf','-nc','-ks','-ss','-sc','-lr','-dst','-sh','-rbf','-sm',
                            '-lrd','-wd','-wm','-p','-t','-pl','-hid','-trs','-tes','-op','-bs'})
opt.save = opt.save:gsub('PARAMS', parsed)

if opt.type == 'float' then
   torch.setdefaulttensortype('torch.FloatTensor')
else
   torch.setdefaulttensortype('torch.DoubleTensor')   
end

if opt.seed then
   torch.manualSeed(opt.seed)
end

if opt.log then
   xlua.log(sys.dirname(opt.save) .. '/session.txt')
end

opt.threads = tonumber(opt.threads)
if opt.threads > 1 then
   torch.setnumthreads(opt.threads)
   print('<trainer> using ' .. opt.threads .. ' threads')
end



opt.quicktest = false -- true = small test, false = full code running 
opt.slacmodel = false -- true = SLAC, false = fully connected layers
opt.cnnmodel = true -- true = convnet model with tanh and normalization, otherwise without
opt.videodata = false -- true = load video file, otherwise siftflow data




----------------------------------------------------------------------
-- define network to train
--

pcall(loadstring("f0_,f1_,f2_,f3_ = " .. opt.nfeatures))         -- nb of features
pcall(loadstring("c0_,c1_,c2_,c3_ = " .. opt.nconnex))           -- fanin for each layer
pcall(loadstring("k0_,k1_,k2_,k3_ = " .. opt.kersize))           -- size of kernels
pcall(loadstring("s0_,s1_         = " .. opt.subsize))           -- size of subsamplers (strides)
pcall(loadstring("scales          = {" .. opt.scales .. "}"))    -- scales
trainable_ss    = s0_*s1_                                        -- global subsampling
trainable_fov   = ((k3_*s1_+k2_-1)*s0_+k1_-1)                    -- global field of view
patchSize       = trainable_fov                                  -- patch size is the same
p0_,p1_         = f3_*#scales,tonumber(opt.hiddens)              -- dimensions for top perceptron

normthres = 1e-1

if not opt.network then
   print('<trainer> creating new network')

   -- Preprocessor (normalizer)
   preproc = nn.Sequential()
   if opt.preproc == 'norm(rgb)' then
      preproc:add(nn.SpatialContrastiveNormalization(f0_, image.gaussian1D(k0_), normthres))
   elseif opt.preproc == 'norm(yuv)' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      preproc:add(nn.SpatialContrastiveNormalization(f0_, image.gaussian1D(k0_), normthres))
   elseif opt.preproc == 'norm(y)+norm(u)+norm(v)' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      do
         normer = nn.ConcatTable()
         for i = 1,3 do
            local n = nn.Sequential()
            n:add(nn.Narrow(1,i,1))
            n:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
            normer:add(n)
         end
      end
      preproc:add(normer)
      preproc:add(nn.JoinTable(1))
   elseif opt.preproc == 'norm(y)+uv' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
      do
         ynormer = nn.Sequential()
         ynormer:add(nn.Narrow(1,1,1))
         ynormer:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
         normer = nn.ConcatTable()
         normer:add(ynormer)
         normer:add(nn.Narrow(1,2,2))
      end
      preproc:add(normer)
      preproc:add(nn.JoinTable(1))
   elseif opt.preproc == 'norm(y)' then
      f0_ = 1
      preproc:add(nn.SpatialColorTransform('rgb2y'))
      preproc:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(k0_), normthres))
   elseif opt.preproc == 'rgb' then
      preproc:add(nn.Identity())
   elseif opt.preproc == 'yuv' then
      preproc:add(nn.SpatialColorTransform('rgb2yuv'))
   else
      print('incorrect arg: preproc')
      op:help()
      os.exit()
   end




----------------------------------------------------------------------
if opt.videodata then
   print '==> loading videt training-set'
   dspath = '../datasets/driving1.mov'
   --source = ffmpeg.Video{path=dspath, width = 120, height = 80, encoding='jpg', fps=24, loaddump=false, load=false}
   source = ffmpeg.Video{path=dspath, width = 320, height = 240, encoding='jpg', fps=24, loaddump=false, load=false}
   
   rawFrame = source:forward()
   -- input video params:
   ivch = rawFrame:size(1) -- channels
   ivhe = rawFrame:size(2) -- height
   ivwi = rawFrame:size(3) -- width
   source.current = 1 -- rewind video frames

else 
   print '==> loading siftflow training-set'
   dofile('sf-dataset.lua')
   
   -- input image dateaset params:
   ivch = trainData[1][1]:size(1) -- channels
   ivhe = trainData[1][1]:size(2) -- height
   ivwi = trainData[1][1]:size(3) -- width
   
end




----------------------------------------------------------------------


   -- Connex tables
   if f0_ == 3 and f1_ == 16 and c1_ == 1 then
      table1 = torch.Tensor{
         {1, 1},{1, 2},{1, 3},{1, 4},
         {1, 5},{1, 6},{1, 7},{1, 8},
         {1, 9},{1,10},{1,11},{1,12},
         {2,13},{2,14},{3,15},{3,16}
      }
   else
      table1 = nn.tables.random(f0_,f1_,c1_)
   end
   table2 = nn.tables.random(f1_,f2_,c2_)
   table3 = nn.tables.random(f2_,f3_,c3_)
   
   
   
   
   -- Trainable Network
   convnet = nn.Sequential()
   convnet:add(nn.SpatialConvolutionMap(table1,k1_,k1_))
   convnet:add(nn.Tanh())
   if opt.pooling == 'max' then
      convnet:add(nn.SpatialMaxPooling(s0_,s0_,s0_,s0_))
   elseif opt.pooling == 'sum' then
      convnet:add(nn.SpatialSubSampling(f1_,s0_,s0_,s0_,s0_))
   elseif opt.pooling == 'l2' then
      convnet:add(nn.SpatialLPPooling(f1_,2,s0_,s0_,s0_,s0_))
   end

   convnet:add(nn.SpatialConvolutionMap(table2,k2_,k2_))
   convnet:add(nn.Tanh())
   if opt.pooling == 'max' then
      convnet:add(nn.SpatialMaxPooling(s1_,s1_,s1_,s1_))
   elseif opt.pooling == 'sum' then
      convnet:add(nn.SpatialSubSampling(f2_,s1_,s1_,s1_,s1_))
   elseif opt.pooling == 'l2' then
      convnet:add(nn.SpatialLPPooling(f2_,2,s1_,s1_,s1_,s1_))
   end

   convnet:add(nn.SpatialConvolutionMap(table3,k3_,k3_))



----------------------------------------------------------------------
--
   -- Replicate convnet:
   convnets = {}
   preprocessors = {}
   for i in ipairs(scales) do
      if opt.share == 'all' then
         table.insert(convnets, convnet:clone('weight','bias','gradWeight','gradBias'))
      elseif tonumber(opt.share) == 0 then
         table.insert(convnets, convnet:clone())
      elseif tonumber(opt.share) > 0 then
         opt.share = tonumber(opt.share)
         local newnet = convnet:clone()
         if opt.share >= 1 then
            newnet.modules[1].weight:set(convnet.modules[1].weight)
            newnet.modules[1].bias:set(convnet.modules[1].bias)
            newnet.modules[1].gradWeight:set(convnet.modules[1].gradWeight)
            newnet.modules[1].gradBias:set(convnet.modules[1].gradBias)
         end
         if opt.share >= 2 then
            newnet.modules[4].weight:set(convnet.modules[4].weight)
            newnet.modules[4].bias:set(convnet.modules[4].bias)
            newnet.modules[4].gradWeight:set(convnet.modules[4].gradWeight)
            newnet.modules[4].gradBias:set(convnet.modules[4].gradBias)
         end
         table.insert(convnets, newnet)
      end
      table.insert(preprocessors, preproc:clone())
   end

   -- Fovea Container
   fovea = nn.SpatialFovea{nInputPlane = 3,
                           ratios = scales,
                           preProcessors = preprocessors,
                           processors = convnets,
                           fov = trainable_fov,
                           sub = trainable_ss,
                           cachePrePreproc = true}

----------------------------------------------------------------------
--
   -- Classifier
   classifier = nn.Sequential()
   if opt.rbf then
      -- rbf classifier
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,p1_))
      classifier:add(nn.Tanh())
      classifier:add(nn.WeightedEuclidean(p1_,#classes))
      classifier:add(nn.Power(-2))
      --classifier:add(nn.Minus()) -- this does not exist...
   elseif p1_ == 0 then
      -- no hidden layers: a simple linear classifier
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,#classes))
   else
      -- a 2-layer perceptron
      classifier:add(nn.Tanh())
      classifier:add(nn.Reshape(p0_))
      classifier:add(nn.Linear(p0_,p1_))
      classifier:add(nn.Tanh())
      classifier:add(nn.Linear(p1_,#classes))
   end

   -- Global trainable machine
   trainable = classifier -- nn.Sequential() -- EC unsup network mods
   --trainable:add(fovea) --train only classifier with Clustering Learning/unsup network
   --trainable:add(classifier) --train only classifier with Clustering Learning/unsup network
   
   -- Save model
   if opt.save then
      print('<trainer> saving bare network to '..opt.save)
      os.execute('mkdir -p "' .. sys.dirname(opt.save) .. '"')
      torch.save(opt.save, trainable)
   end
else
   -- reload network
   print('<trainer> reloading previously trained network')
   trainable = torch.load(opt.network)

   -- get pointers to internal modules
   fovea = trainable.modules[1]
   classifier = trainable.modules[2]
   convnet = fovea.processors[1]

   -- optionally replace classifier?
   if p1_ ~= 0 and #classifier.modules < 5 then
      -- replacing linear classifier by 2-layer perceptron
      print('<trainer> dropping linear classifier, replacing by 2-layer perceptron')
      classifier.modules[3] = nil
      classifier:add(nn.Linear(p0_,p1_))
      classifier:add(nn.Tanh())
      classifier:add(nn.Linear(p1_,#classes))
   end
end

-- retrieve parameters and gradients
parameters,gradParameters = trainable:getParameters()

-- verbose
print('<trainer> model:')
print(trainable)
print('<trainer> nb of trainable parameters: ' .. parameters:size(1))

----------------------------------------------------------------------
-- training criterion: Mean-Square Error or Negative Log-Likelihood
--
criterion = nn.DistNLLCriterion()
criterion.targetIsProbability = true



----------------------------------------------------------------------
-- load/get dataset

--print '==> loading siftflow training-set'
--dofile('sf-dataset.lua')




----------------------------------------------------------------------
-- process images in dataset with unsupervised network 'tnet':
--

--trainsize = 200
--testsize = 200
trainsize = trainData:size()
testsize = testData:size()

in_trsz = trainData[1][1]:size(2)
in_tesz = testData[1][1]:size(2)
--if nnf1 >1 then
--   out_trsz = tnet:forward(trainData[1][1]:resize(ivch,1,in_trsz,in_trsz):expand(ivch,nnf1,in_trsz,in_trsz))
--   out_tesz = tnet:forward(testData[1][1]:resize(ivch,1,in_tesz,in_tesz):expand(ivch,nnf1,in_tesz,in_tesz))
--elseif nnf1 ==1 then
--   out_trsz = tnet:forward(trainData[1][1]:resize(ivch,in_trsz,in_trsz))
--   out_tesz = tnet:forward(testData[1][1]:resize(ivch,in_tesz,in_tesz))
--end




-------------------------
-- free up memory, reset fovea?

print "==> processing dataset with videoknet:"
trainData2 = {}
for t = 1,trainsize do
   trainData2[t] = {}
   local sample = trainData[t]
   local input = sample[1]
   if input:size(1) == 3 then
      local target = sample[2]
      local sample_x = sample[3]
      local sample_y = sample[4]
      local sample_size = sample[5]
      fovea:focus(sample_x, sample_y, sample_size)
      trainData2[t][1] = fovea:forward(input):clone()
      trainData2[t][2] = target
   end 
   xlua.progress(t, trainsize)
end

--report some statistics:
print('trainData2[1][1] Max: '..trainData2[1][1]:max()..' and min: '..trainData2[1][1]:min()..' and mean: '..trainData2[1][1]:mean())


testData2 = {}
fovea:focus() -- defocus to process all dataset images in full size~
for t = 1, #testData.rawdata/5 do
   testData:loadSample((t-1)*5+1)
   testData2[t] = {}
   local input = testData.currentSample
   testData2[t][1] = fovea:forward(input):clone()
   testData2[t][2] = testData.currentMask:clone()
   xlua.progress(t, #testData.rawdata/5)
end

--report some statistics:
print('testData2[1][1] Max: '..testData2[1][1]:max()..' and min: '..testData2[1][1]:min()..' and mean: '..testData2[1][1]:mean())




-------------------------
-- do training:
dofile('sf-vkn-train.lua')


