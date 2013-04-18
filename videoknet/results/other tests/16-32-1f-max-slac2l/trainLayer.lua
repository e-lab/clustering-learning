----------------------------------------------------------------------
-- Massive online trained network on videos
-- videoknet functions
-- E. Culurciello, Feb 1st 2013
--

function trainLayer(nlayer,trainData, nsamples, kernels, nk, nnf, is)
   -- input video params:
   local ivch = trainData[1]:size(1) -- channels
   local ivhe = trainData[1]:size(2) -- height
   local ivwi = trainData[1]:size(3) -- width
   
   print '==> extracting patches' -- only extract on Y channel (or R if RGB) -- all ok
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
      xlua.progress(i, nsamples)
   end
   
   -- show a few patches:
   if opt.visualize then
      f256S = data[{{1,256}}]:reshape(256,nnf*is,is)
      image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches of video frames'}
   end
   
   ----------------------------------------------------------------------
   print '==> running k-means to learn filter'
   local win
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk,nnf*is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
      end
   end                    
   --kernels = kmec(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true) -- Euge kmeans (not good init yet)
   --kernels = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.batchsize, cb, true)
   kernels = okmeans(data, nk, kernels, opt.initstd, opt.niter, opt.batchsize, cb, true) -- online version to upadte filters
   print('==> saving centroids to disk:')
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
   win = image.display{image=kernels:reshape(nk,nnf*is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
   
   print '==> verify filters statistics'
   print('filters max mean: ' .. kernels:mean(2):abs():max())
   print('filters max standard deviation: ' .. kernels:std(2):abs():max())
         
   -- save animated GIF of filters: 
   if nlayer == 1 then
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




