----------------------------------------------------------------------
-- Massive online trained network on videos
-- videoknet functions
-- E. Culurciello, Feb 1st 2013
--

function trainLayer(nlayer, invdata, nsamples, kernels, nk, nnf, is, verbose)
   -- computes filter kernels for a layer with Clustering Learning / k-means
   -- verbose = true ==> show images, text messages
  
   -- input video params:
   local ivch = invdata[1]:size(1) -- channels
   local ivhe = invdata[1]:size(2) -- height
   local ivwi = invdata[1]:size(3) -- width
   
   local patchminstd = 0--.25 -- min std require to use patch! IMPORTANT PARAM!!!
   
   if verbose then  print '==> extracting patches' end
   local img = torch.Tensor(ivch, nnf, ivhe, ivwi)
   local data = torch.Tensor(nsamples, ivch*nnf*is*is) -- need to learn volumetric filters on multiple frames!
   local i = 1
   while i <= nsamples do
      fimg = math.random(nnf,invdata:size(1)) -- pointer to current frame
      for j = 1, nnf do
         img[{{},{j}}] = invdata[fimg-j+1] -- pointer to current and all previous frames
      end
      local z = math.random(1,ivch)
      local x = math.random(1,ivwi-is+1)
      local y = math.random(1,ivhe-is+1)
      local patches = img[{ {},{},{y,y+is-1},{x,x+is-1} }]:clone()
      -- keep only patches with high SNR:
      data[i] = patches
      i = i+1 -- if patches is used then count up
      if verbose then xlua.progress(i, nsamples) end
   end
   
   -- show a few patches:
   if verbose then
      f256S = data[{{1,256}}]:reshape(256,ivch,nnf*is,is)
      image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches of input data before clustering'}
   end
  
   ----------------------------------------------------------------------
   require 'unsup'
   function zca_whiten(x)          -- zca whitening is used
      local dims = x:size()        -- outputs are whitened data(x),
      local nsamples = dims[1]     -- M, P statistic (matrix)
      local ndims    = dims[2]
      local M = torch.mean(x, 1)
      local D, V = unsup.pcacov(x)
      x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
      local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
      local P = V * diag * V:t()
      x = x * P
      return x, M, P
   end
   Mmat = torch.Tensor()           -- to avoid error if no whitening
   Pmat = torch.Tensor()           -- to avoid error if no whitening

   -- apply whitening only for 1st input
   if nlayer == 1 and opt.whitening then
      if verbose then print '==> applying whitening to input patches' end
      data, Mmat, Pmat = zca_whiten(data)
   end

   ----------------------------------------------------------------------
   if verbose then print '==> running k-means to learn filter' end
   
   local win
   function cb (kernels)
      if verbose then
         win = image.display{image=kernels:reshape(nk,ivch, nnf*is, is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
      end
   end                    
   
   --local kernels = torch.Tensor(nk, ivch, nnf, is, is)
   --local counts = torch.Tensor(nk, ivch, 1)
   --kernels = kmec(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true) -- Euge kmeans (not good init yet)
   --kernels = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true)
   kernels, counts = okmeans(data, nk, nil, opt.initstd, opt.niter, opt.batchsize, cb, true)
--   for i = 1, ivch do
--      kernelss, countss = okmeans(data[{{},{i}}]:reshape(nsamples,nnf*is*is), nk, nil, opt.initstd, opt.niter, opt.batchsize, cb, verbose) -- online version to upadte filters
--      kernels[{{},{i}}] = kernelss:reshape(nk,1,nnf, is, is)
--      counts[{{},{i}}] = countss:reshape(nk,1,1)
--   end
   
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
      win = image.display{image=kernels:reshape(nk, ivch, nnf*is, is), padding=2, symmetric=true, 
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
  
   return kernels, counts, Mmat, Pmat  -- return M, P matrix for whitening
end



function processLayer(lv, network, data_in, nkernels, oheight, owidth, verbose)
   data_out = torch.Tensor(data_in:size(1), nkernels, oheight, owidth)
   for i = nnf1, data_in:size(1) do -- just get a few frames to begin with
      procFrames = data_in[i]
      data_out[i] = network:forward(procFrames)
      xlua.progress(i, data_in:size(1))
      -- do a live display of the input video and output feature maps 
      if verbose then
         winm = image.display{image=data_out[i], padding=2, zoom=1, win=winm, nrow=math.floor(math.sqrt(nkernels))}
      end
   end
   -- data_out = nil --free memory if needed
   return data_out
end



function createCoCnx(nlayer, vdata, nkp, fpgroup, fanin, samples, nnf, is, prev_ker, verbose)
   
   assert(nkp == vdata:size(2), 'Error: nkp and input video features are not the same number!') -- number features
   local nf = vdata:size(1)

   local vd1 = torch.zeros(vdata:size(1),fanin,vdata:size(3),vdata:size(4)) --temp data storage
   
   local covMat = torch.zeros(nkp,nkp) --covariance matrix
   local conMat = torch.zeros(nkp,nkp)
   local connTable = {} -- table of connections
   local kerTable = {} -- table of kernels/filters
      
   -- compute covariance matrix:
   for k=1,nf do         -- 1
      for i=1,nkp do      -- numKernels
         for j=i,nkp do   -- numKernels
            covMat[i][j] = covMat[i][j]+torch.dot(vdata[k][i],vdata[k][j])/(torch.sum(torch.pow(vdata[k][i],2))*torch.sum(torch.pow(vdata[k][j],2))) -- dist metric 
            covMat[j][i] = covMat[i][j]
         end
      end
   end   
   
   -- connect cells in fanin groups:
   for i=1,nkp do
      max, inx = torch.sort(covMat[i],true) --want smaller values first (dist)
      group = torch.Tensor(fanin)
      -- groups of fanin (connect top two max):
      m=1
      group[1] = i
      vd1[{{},{1}}] = vdata[{{},{i}}]
      conMat[i][i]=1
      for k=2,fanin do -- the first value connects to itself!
         while (conMat[i][inx[m]]==1) do 
           print('m', m, conMat[i][inx[m]])
           m=m+1
         end
         print('inx[m]', inx[m], 'm', m)
         group[k] = inx[m]
         vd1[{{},{k}}] = vdata[{{},{inx[m]}}]
         conMat[i][inx[m]]=1
         conMat[inx[m]][i]=1
      end
      for j=1, fpgroup do 
         for k=1, fanin do
            table.insert(connTable, torch.Tensor({group[k],(i-1)*fpgroup+j}))
         end
      end

      kerp = trainLayer(nlayer, vd1, samples, nil, fpgroup, nnf, is, verbose)
      for j=1,fpgroup do
         for k=1,fanin do
            table.insert(kerTable, kerp:reshape(fpgroup, fanin,is*is)[j][k])
         end
      end
      
      if verbose then xlua.progress(i, nkp) end
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
   
   --renormalize all kernels:
   for i=1,kerTensor:size(1) do
      kerTensor[i] = kerTensor[i]:add(-kerTensor[i]:mean()):div(kerTensor[i]:std())
   end
   
   return connTableTensor, kerTensor
end


-- tests:
-- image.display{image=kernels1:reshape(32,7,7), padding=2, zoom=4,  nrow=8}
-- image.display{image=videoData2[6], padding=2, zoom=1,  nrow=8}




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




