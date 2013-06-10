
----------------------------------------------------------------------
-- Authors: Eugenio Culurciello, Aysegul Dundar, Jordan Bates, Jonghoon Jin
-- Functions : 
-- 1 ) trainLayer : Extract random pathces for kmeans
-- train the filters with kmenas
-- 2) processLayer : Processed the data with the network and 
-- normalized the output
-- 3 ) createCoCnx : creates connection table and train filters 
-- for each connection, details: http://arxiv.org/abs/1306.0152
----------------------------------------------------------------------

function trainLayer(nlayer, invdata, nsamples, nk, is, verbose)
   -- input video params:
   local ivch = invdata[1]:size(1) -- channels
   local ivhe = invdata[1]:size(2) -- height
   local ivwi = invdata[1]:size(3) -- width
      
   if verbose then  print '==> extracting patches' end
   local img = torch.Tensor(ivch, 1, ivhe, ivwi)
   local data = torch.Tensor(nsamples, ivch*is*is) 
   local i = 1
   while i <= nsamples do
      fimg = math.random(1,invdata:size(1)) -- pointer to current frame    
      img[{{},{1}}] = invdata[fimg] -- pointer to current and all previous frames
 
      local z = math.random(1,ivch)
      local x = math.random(1,ivwi-is+1)
      local y = math.random(1,ivhe-is+1)
      local patches = img[{ {},{},{y,y+is-1},{x,x+is-1} }]:clone()   
      data[i] = patches
      i = i+1 -- if patches is used then count up   
      if verbose then xlua.progress(i, nsamples) end
   end 

   Mmat = torch.Tensor()           -- to avoid error if no whitening
   Pmat = torch.Tensor()           -- to avoid error if no whitening

   -- apply whitening only for 1st layer input
   if nlayer == 1 and opt.whitening then
      if verbose then print '==> applying whitening to input patches' end
      data, Mmat, Pmat = zca_whiten(data)
   end     
   ----------------------------------------------------------------------
   if verbose then print '==> running k-means to learn filter' end                   
   kernels, counts = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, verbose)

   for i=1,nk do
      -- normalize kernels to 0 mean and 1 std:
      kernels[i]:add(-kernels[i]:mean())
      kernels[i]:div(kernels[i]:std())
       -- clear nan kernels   
      if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
         print('Found NaN kernels!') 
         kernels[i] = torch.zeros(kernels[1]:size()) 
      end
   end
   -- display final filters:
   if verbose then
      win = image.display{image=kernels:reshape(nk, ivch, 1*is, is), padding=2, symmetric=true, 
         zoom=4, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
   end     
   return kernels, counts, Mmat, Pmat 
end



function processLayer(network, traindata, testdata, verbose)
   a = network:forward(traindata[1])

   train_out = torch.Tensor(trsize, a:size(1), a:size(2), a:size(3))
   test_out  = torch.Tensor(tesize, a:size(1), a:size(2), a:size(3))

   for i = 1, trsize do
      train_out[i] = network:forward(traindata[i])      
      xlua.progress(i, traindata:size(1))
   end
   for i = 1, tesize do
      test_out[i] = network:forward(testdata[i])      
      xlua.progress(i, testdata:size(1))
   end
   -- Normalize the output
   mean = {}
   std = {}

   for i=1, train_out:size(2) do
      -- normalize each channel globally:
       mean[i] = train_out[{ {},i,{},{} }]:mean()
       std[i] = train_out[{ {},i,{},{} }]:std()
       train_out[{ {},i,{},{} }]:add(-mean[i])
       train_out[{ {},i,{},{} }]:div(std[i])
       test_out[{ {},i,{},{} }]:add(-mean[i])
       test_out[{ {},i,{},{} }]:div(std[i])
    end

   return train_out, test_out
end



-- random conn version Ayse, JY, May 2013
function createCoCnx(nlayer, vdata, nkp, fpgroup, fanin, samples, is, verbose)
   -- create a covariance/co-occurence connection table based on some test data
   -- input data has multiple planes, compute similarity between these planes
   -- group planes that are similar
   -- we connect each group to one neuron in next layer
   -- nkp = features previous layer, fpgroup = feateures per group
   -- fanin = desired connex fanin - this should be also learned from data...
   
   -- train filter for next layer based on groups of connTable!!!
   -- uses co-occurence of features on muliple maps: sum maps, run clustering on them
    
   assert(nkp == vdata:size(2), 'Error: nkp and input video features are not the same number!') -- number features
   
   local nf = vdata:size(1) --number frames
   local vd1 = torch.zeros(vdata:size(1),fanin,vdata:size(3),vdata:size(4)) --temp data storage
   local covMat = torch.zeros(nkp,nkp) --covariance matrix
   local connTable = {} -- table of connections
   local kerTable = {} -- table of kernels/filters
   
   -- connect cells in fanin groups:
   for i=1,nkp do 
      -- groups of fanin (connect top two max):
      for k=1,fanin do -- the first value connects to itself!
            inx = math.random(1,nkp)
            -- group all feature maps that co-occur / are similar
            vd1[{{},{k}}] = vdata[{{},{inx}}]
      end
      kerp = trainLayer(nlayer, vd1, samples, fpgroup, is, verbose)
      for j=1,fpgroup do
         for k=1,fanin do
             table.insert(kerTable, kerp:reshape(fpgroup, fanin,is*is)[j][k])
             table.insert(connTable, torch.Tensor({inx,(i-1)*fpgroup+j}))
         end
      end
      
      xlua.progress(i, nkp)
   end
 
   -- turn tables into tensors:
   local connTableTensor = torch.Tensor(#connTable,2)
   local kerTensor = torch.zeros(#kerTable,is,is)
   for i, value in ipairs(connTable) do
      connTableTensor[i] = value
   end
   for i, value in ipairs(kerTable) do
      kerTensor[i] = value
   end
   
   return connTableTensor, kerTensor
end



