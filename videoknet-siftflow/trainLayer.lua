----------------------------------------------------------------------
-- Massive online trained network on videos
-- videoknet functions
-- E. Culurciello, Feb 1st 2013
--

function trainLayer(nlayer,trainData, nsamples, kernels, nk, nnf, is, verbose)
   -- computes filter kernels for a layer with Clustering Learning / k-means
   -- verbose = true ==> show images, text messages
   
   -- input video params:
   local ivch = trainData[1]:size(1) -- channels
   local ivhe = trainData[1]:size(2) -- height
   local ivwi = trainData[1]:size(3) -- width
   
   if verbose then print '==> extracting patches' end -- only extract on Y channel (or R if RGB) -- all ok
   local img = torch.Tensor(ivch, nnf, ivhe, ivwi)
   local data = torch.Tensor(nsamples, nnf*is*is) -- need to learn volumetric filters on multiple frames!
   for i = 1, nsamples do
      fimg = math.random(nnf,nfpr) -- pointer to current frame
      for j = 1, nnf do
         img[{{},{j}}] = trainData[fimg-j+1] -- pointer to current and all previous frames
      end
      local z = math.random(1,ivch)
      local x = math.random(1,ivwi-is+1)
      local y = math.random(1,ivhe-is+1)
      local patches = img[{ {z},{},{y,y+is-1},{x,x+is-1} }]:clone()
      patches:add(-patches:mean())
      patches:div(patches:std()+1e-3) -- to prevent divide-by-0
      -- TODO: keep only patches with high SNR?
      data[i] = patches
      if verbose then xlua.progress(i, nsamples) end
   end
   
   -- show a few patches:
   if verbose then
      f256S = data[{{1,256}}]:reshape(256,nnf*is,is)
      image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches of video frames'}
   end
   
   ----------------------------------------------------------------------
   if verbose then print '==> running k-means to learn filter' end
   local win
   function cb (kernels)
      if verbose then
         win = image.display{image=kernels:reshape(nk,nnf*is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
      end
   end                    
   --kernels = kmec(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true) -- Euge kmeans (not good init yet)
   --kernels = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true)
   kernels = okmeans(data, nk, kernels, opt.initstd, opt.niter, opt.batchsize, cb, verbose) -- online version to upadte filters
   if verbose then print('==> saving centroids to disk:') end
   --torch.save('volumetric.t7', kernels)
   
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
   -- print final filters:
   if verbose then
      win = image.display{image=kernels:reshape(nk,nnf*is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
      print '==> verify filters statistics'
      print('filters max mean: ' .. kernels:mean(2):abs():max())
      print('filters max standard deviation: ' .. kernels:std(2):abs():max())
   end
         
   -- save animated GIF of filters: 
   if nlayer == 1 and display_yo then
      for i=1, nnf do
         outf = image.display{image=kernels:reshape(nk,nnf,is,is)[{{},{i},{},{}}]:reshape(nk,is,is), 
            symmetric=true, padding=2, zoom=2, offscreen=true, nrow=math.floor(math.sqrt(nk))}
         outima = outf.image(outf)
         outima.save(outima, "volumetric"..i..".png")
      end
      -- use imagemagick (brew package) to convert into a Animated GIF file:
      os.execute('convert -delay 20 -loop 0 volumetric*.png volumetric.gif')
      os.execute('rm volumetric*.png') -- remove intermediate files
   end
   
   return kernels
end



function createCoCnxTable(vdata, nkp, nkn, fanin, mode)
   -- create a covariance/co-occurence connection table based on some test data
   -- input data has multiple planes, compute similarity between these planes
   -- group planes that are similar
   -- we connect each group to one neuron in next layer
   -- mode = 'OR', 'AND'. OR=distance similarity metric, AND=co-occurrence metric
   -- nkp = features previous layer, nkn = next layer
   -- fanin = desired connex fanin - this should be also learned from data...
   
   local nf = vdata:size(1) --number frames
   assert(nkp == vdata:size(2), 'Error: nkp and input video features are not the same number!') -- number features
   
   local covMat = torch.zeros(nkp,nkp) --covariance matrix
   local connTable = {}
   
   for k=1,nf do
      for i=1,nkp do
         for j=i+1,nkp do
            if mode == 'OR' then covMat[i][j] = covMat[i][j] + torch.dist(vdata[k][i], vdata[k][j]) -- dist metric
            elseif mode == 'AND' then covMat[i][j] = covMat[i][j] + torch.cmul(vdata[k][i], vdata[k][j]):sum() -- conv metric
            else print('Error: mode must be AND or OR!') end
            covMat[j][i] = covMat[i][j]
         end
      end
   end
   -- current version only computes this number of connex max:
   local nmaxconnex = nkp*fanin+nkp*fanin*2
   -- connect cells in fanin groups:
   for i=1,nkp do
      if mode == 'OR' then max, j = torch.sort(covMat[i]) --want smaller values first (dist)
      else max, j = torch.sort(covMat[i], true) end -- want larger values first (conv)
      for k=1,fanin do -- the first value may connect to itself!
         table.insert(connTable, torch.Tensor({j[k],i}))
      end
   end
   -- connect cells in fanin*2 groups:
   for i=1,nkp do
      if mode == 'OR' then max, j = torch.sort(covMat[i]) --want smaller values first (dist)
      else max, j = torch.sort(covMat[i], true) end -- want larger values first (conv)
      for k=1,fanin*2 do -- the first value may connect to itself!
         table.insert(connTable, torch.Tensor({j[k],i+nkp}))
      end
   end
   
   -- turn table into connect Table tensor, as needed by nn modules
   connTableTensor = torch.Tensor(#connTable,2)
   for i, value in ipairs(connTable) do
      connTableTensor[i] = value
   end
   return connTableTensor[{{1, nmaxconnex}}]
end


-- tests:
-- image.display{image=kernels1:reshape(32,7,7), padding=2, zoom=4,  nrow=8}
-- image.display{image=videoData2[1], padding=2, zoom=1,  nrow=8}



function trainCoCnxLayer(nlayer, vdata, connTable, samples, nk, fanin, nnf, is, verbose)
   -- train filter for next layer (kernels2) based on groups of cTable!!!
   -- uses co-occurence of features on muliple maps: sum maps, run clustering on them
   
   -- USAGE: run createCoCnxTable first, this then creates filters based on connex 
   -- used in that function!!!
   
   -- nnf = number frames, nk = number kernels, is = kernel size
   -- verbose = true ==> show images, text messages
   
   local vd2 = torch.zeros(nfpr,1,vdata:size(3),vdata:size(4))
   kernels = torch.zeros(connTable:size(1),is*is)
   
   -- ATTENTION: this below should really be rewritten to scan connTable... 
   
   -- connect cells in fanin groups:
   for i=1,nk/fanin do
      --kernels2[] = trainlayer on videoData features only from the group!!
      for j=1,fanin do -- only do kmeans on features of group:
         -- sum up all feature maps that co-occur (AND operation)
         vd2 = vd2 + vdata[{{},{connTable[(i-1)*fanin+j][1]}}]
      end
      -- learn one filter for this connection:
      kerp = trainLayer(nlayer, vd2, samples, nil, 1, nnf, is, verbose)
      --replicate kernels to all group
      kernels[{{(i-1)*fanin+1, i*fanin}}] = kerp:reshape(1,is*is):expand(fanin,is*is)
   end
   -- connect cells in fanin*2 groups:
   for i=nk/fanin+1,3*nk/fanin,2 do
      --kernels2[] = trainlayer on videoData features only from the group!!
      for j=1,fanin*2 do -- only do kmeans on features of group:
         -- sum up all feature maps that co-occur (AND operation)
         vd2 = vd2 + vdata[{{},{connTable[(i-1)*fanin+j][1]}}]
      end
      -- learn one filter for this connection:
      kerp = trainLayer(nlayer, vd2, samples, nil, 1, nnf, is, verbose)
      --replicate kernels to all group
      kernels[{{(i-1)*fanin+1,(i+1)*fanin}}] = kerp:reshape(1,is*is):expand(fanin*2,is*is)
   end
   return kernels
end



--function createConnexTable(nkP, nkN, level)
--   -- computes a connection matrix LeNet5-style:
--   -- each higher layer neurons are connected to 1,2,4,8,16... in lower level 
--   -- nkP = lower layer features #
--   -- nkN = higher layer feature #
--   -- level = fanin level:
--   -- level 1,2,3,4.. means each nkN gets fanin of 2,4,8,16..
--   -- fanin units are randomized (might not be best option!!!)
--   table_size = 0
--   indexL = 1
--   indexH = 0
--   for i=1,level do
--      table_size = table_size + 2^i*nkN
--   end
--   connTable = torch.Tensor(table_size,2)
--   for i=1,level do
--      indexH = indexH + nn.tables.random(nkP,nkN,2^i):size(1)
--      connTable[{{indexL,indexH}}]=nn.tables.random(nkP,nkN,2^i)
--      indexL = indexL + nn.tables.random(nkP,nkN,2^i):size(1)
--   end
--   return connTable
--end




