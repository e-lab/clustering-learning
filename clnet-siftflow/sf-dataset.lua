
----------------------------------------------------------------------
-- Classes to learn

classes = {'unknown',
           'awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus',
           'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field',
           'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river',
           'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky',
           'staircase', 'streetlight', 'sun', 'tree', 'window'}

--ec mod:
--classes = {'unknown',
--				'building','car', 'person',
--           'road','sidewalk', 'sign',}


----------------------------------------------------------------------
-- add jitter to dataset
--
if opt.jitter then
   print('<trainer> applying (or removing) jitter')
   transforms = {'-hflip','-rotate 4','-rotate -4'}
   for _,trans in ipairs(transforms) do
      os.execute('torch -ng expand.lua ' .. trans .. ' -d ' .. opt.dataset)
   end
end

----------------------------------------------------------------------
-- load/get dataset
--

if not sys.dirp(opt.dataset) then
   print('<trainer> retrieving dataset')
   local path = sys.dirname(opt.dataset)
   local tar = sys.basename(opt.www)
   os.execute('mkdir -p "' .. path .. '"; '..
              'cd "' .. path .. '"; '..
              'wget ' .. opt.www .. '; '..
              'tar xvf ' .. tar)
elseif opt.clearcache then
   print('<trainer> clearing dataset cache')
   os.execute('rm ' .. opt.dataset .. '/*/subset*/cached*')
end

-- live display
disp = {}
livedisp = function(full_sample, full_mask, sample, ctr_target, x, y, size)
              if fovea.padded[1] then
                 disp.win1=image.display{image=fovea.padded[1], win=disp.win1,
                                         legend='normed [' .. disp.class .. ']'}
                 disp.win1.painter:setcolor(1,0,0)
                 disp.win1.painter:rectangle(disp.x, disp.y, disp.size, disp.size)
                 disp.win1.painter:stroke()
                 disp.win2=image.display{image=fovea.narrowed[1], win=disp.win2,
                                         legend='focused [' .. disp.class .. ']'}
                 if fovea.narrowed[2] then
                    disp.win3=image.display{image=fovea.narrowed[2], win=disp.win3,
                                            legend='focused [' .. disp.class .. ']'}
                 end
                 if fovea.narrowed[3] then
                    disp.win4=image.display{image=fovea.narrowed[3], win=disp.win4,
                                            legend='focused [' .. disp.class .. ']'}
                 end
              end
              disp.x = x; disp.y = y; disp.size = size; disp.class = classes[ctr_target]
              sys.sleep(1)
           end

-- create a distribution of classes in a given patch
nclasses = #classes
distribution = function(mask)
                  local hist = lab.histc(mask,nclasses,0.5,nclasses+0.5,true)
                  hist:div(hist:sum())
                  return hist
               end

-- this function generates {sample + target}
labelGenerator = function(dataset, full_sample, full_mask, sample, mask, ctr_target,
                          ctr_x, ctr_y, box_x, box_y, box_size)
                    -- distort?
                    if opt.distort then
                       full_sample,full_mask = distort(full_sample,full_mask)
                    end
                    -- generate target vector
                    local target
                    if opt.distributions then
                       target = distribution(mask)
                    else
                       target = torch.Tensor(#classes):zero()
                       target[ctr_target] = 1
                    end
                    -- display sample
                    if opt.display then 
                       livedisp(full_sample, full_mask, sample, ctr_target, ctr_x, ctr_y, box_size) 
                    end
                    -- return
                    return {full_sample, target, ctr_x, ctr_y, box_size}
                 end

-- sampling filter: only sample patches that have at least N% pixels of the class
local filter = {ratio=0.1, size=25, step=4}

-- distort function
function distort(i,t)
   -- bernoulli
   if torch.bernoulli(0.9) == 1 then
      return i,t
   end

   -- x/y grids
   local grid_y = torch.ger( torch.linspace(-1,1,t:size(1)), torch.ones(t:size(2)) )
   local grid_x = torch.ger( torch.ones(t:size(1)), torch.linspace(-1,1,t:size(2)) )

   -- distort field
   local gsize = 50
   local g1 = image.gaussian1D(gsize):resize(gsize,1)
   local g2 = g1:t()
   local flow1 = image.convolve(torch.rand(2,t:size(1),t:size(2)):add(-0.5), g1, 'same')
   flow1 = image.convolve(flow1, g2, 'same')
   flow1:mul(torch.uniform(0,0.05))

   -- scale field
   local flow2 = torch.Tensor(2,t:size(1),t:size(2))
   flow2[1] = grid_y
   flow2[2] = grid_x
   flow2[1]:mul(torch.uniform(-30,30))
   flow2[2]:mul(torch.uniform(-30,30))

   -- rotation field
   flow3 = torch.Tensor(2,t:size(1),t:size(2))
   flow3[1] = grid_y * ((t:size(1)-1)/2) * -1
   flow3[2] = grid_x * ((t:size(2)-1)/2) * -1
   view = flow3:reshape(2,t:size(1)*t:size(2))
   local function rmat(deg)
      local r = deg/180*math.pi
      return torch.Tensor{{math.cos(r), -math.sin(r)}, 
                          {math.sin(r), math.cos(r)}}
   end
   flow3r = torch.mm(rmat( torch.uniform(-10,10) ), view)
   flow3 = flow3 - flow3r:reshape( 2, t:size(1), t:size(2) )

   -- apply field
   local flow = flow2 + flow3
   local it = image.warp(i,flow,'bilinear')
   local tt = image.warp(t,flow,'simple')

   -- return transformed image/target
   return it,tt
end



----------------------------------------------------------------------
-- train using SGD
--
batchSize = opt.batchSize

trainConfusion = optim.ConfusionMatrix(classes)
testConfusion  = optim.ConfusionMatrix(classes)
logger         = optim.Logger(sys.dirname(opt.save) .. '/log.txt')

-- train data
trainData = DataSetLabelMe{path=sys.concat(opt.dataset,'train'),
                           verbose=true,
                           rawSampleMaxSize=256,
                           nbClasses=#classes,
                           classNames=classes,
                           classToSkip=1,
                           samplingMode=opt.sampling,
                           samplingFilter=filter,
                           infiniteSet=true,
                           labelGenerator=labelGenerator,
                           cacheFile='cached-256-'..patchSize..'-'..opt.type,
                           nbPatchPerSample=5,
                           preloadSamples=false,
                           patchSize=patchSize}

-- load test set
testData = DataSetLabelMe{path=sys.concat(opt.dataset,'test'),
                          verbose=true,
                          nbClasses=#classes,
                          rawSampleMaxSize=256,
                          classNames=classes,
                          classToSkip=1,
                          cacheFile='cached-256-'..patchSize..'-'..opt.type,
                          preloadSamples=false,
                          patchSize=patchSize}


-- display set
if opt.display then
   trainData:display{title='train set'}
   testData.colormap = trainData.colormap
   testData:display{title='test set'}
end