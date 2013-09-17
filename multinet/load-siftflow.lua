require 'nnx'
require 'sys'
require 'image'
require 'LabelMe'

local classes = {'unknown',
                 'awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus',
                 'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field',
                 'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river',
                 'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky',
                 'staircase', 'streetlight', 'sun', 'tree', 'window'}

local nclasses
if opt.smallSift then 
   nclasses = {'unknown', 'building', 'bus', 'car', 'grass', 'person', 'road',  
               'sign', 'sky', 'tree'}
   l = torch.range(1,#nclasses)
   classToSkip = {1,2,3,4,5,6,8,10,11,12,13,14,15,17,18,20,21,22,24,25,26,27,30,31,32,34}
   torch.save('siftflowLabels.t7',nclasses)
else 
   l = torch.range(1,#classes)
   classToSkip = {1}
   torch.save('siftflowLabels.t7',classes)
end

opt = opt or {}
opt.dataset = '../datasets/siftflow_dataset'
opt.sampling = 'equal'
--opt.type = 'double'
opt.www = 'http://data.neuflow.org/data/siftflow_dataset.tgz'
local filter = {ratio=0.1, size=25, step=4}
patchSize = 46

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

-- train data
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
                       if opt.smallSift then
                           target = torch.Tensor(#nclasses):zero()
                           if (ctr_target==7) then  target[2] = 1 end
                           if (ctr_target==9) then  target[3] = 1 end
                           if(ctr_target==16) then  target[4] = 1 end
                           if(ctr_target==19) then  target[5] = 1 end
                           if(ctr_target==23) then  target[6] = 1 end
                           if(ctr_target==28) then  target[7] = 1 end
                           if(ctr_target==29) then  target[8] = 1 end
                           if(ctr_target==33) then  target[9] = 1 end
                       else
                          target = torch.Tensor(#classes):zero()
                          target[ctr_target] = 1
                       end
                    end
                    -- display sample
                    if opt.display then 
                       livedisp(full_sample, full_mask, sample, ctr_target, ctr_x, ctr_y, box_size) 
                    end
                    -- return
                    return {full_sample, target, ctr_x, ctr_y, box_size}
                 end

trainData = nn.LabelMe{path=sys.concat(opt.dataset,'train'),
                           verbose=true,
                           rawSampleMaxSize=256,
                           nbClasses=#classes,
                           classNames=classes,
                           classToSkip=classToSkip,
                           samplingMode=opt.sampling,
                           samplingFilter=filter,
                           infiniteSet=true,
                           labelGenerator=labelGenerator,
                           cacheFile='cached-256-'..patchSize..'-'..'double',--opt.type,
                           nbPatchPerSample=5,
                           preloadSamples=false,
                           patchSize=patchSize}

-- load test set
testData = nn.LabelMe{path=sys.concat(opt.dataset,'test'),
                          verbose=true,
                          nbClasses=#classes,
                          rawSampleMaxSize=256,
                          classNames=classes,
                          classToSkip=classToSkip,
                          cacheFile='cached-256-'..patchSize..'-'..'double',--opt.type,
                          preloadSamples=false,
                          patchSize=patchSize}

trainData.data = torch.Tensor(1,3,46,46)
testData.data = torch.Tensor(1,3,46,46)
print '==> Limiting the testing dataset to 5000 samples'
testData.size = function() return 5000 end -- instad of 20k
