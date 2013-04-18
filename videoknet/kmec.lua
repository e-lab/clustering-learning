--
-- E. Culurciello - January 2013
-- The Clustering algorithm EC version: one that you CAN understand!
--
--   > x: is supposed to be an MxN matrix, where M is the nb of samples and each sample is N-dim
--   > k: is the number of kernels
--   > niter: the number of iterations
--   > batchsize: the batch size [large is good, to parallelize matrix multiplications]
--   > callback: optional callback, at each iteration end
--   > verbose: prints a progress bar...
--
--   < returns the k means (centroids)
--
function kmec(x, k, std, niter, batchsize, callback, verbose)
   -- args
   --batchsize = batchsize or 1000
   --std = std or 0.1

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]
   
   -- init variables:
   local dists = torch.Tensor(k)
   local kernels = torch.Tensor(k, ndims)
   for t = 1, k do
      kernels[t] = x[nsamples-t] -- just pick some random inputs as init kernels
   end
   --kernels = torch.randn(k,ndims)*std -- or... pick random init kernels (standard in kmeans)
   local kernels_avg = torch.zeros(k)
   
   for t = 1, niter do   
      for t = 1, nsamples do
          
         -- get next patch
         local sample = x[t]
      
         -- try to match new sample with kernels
         for i = 1, k do
            dists[i] = torch.dist(sample, kernels[i], 1) -- L1 distance (SAD)
         end
      
         -- closest template:
         local min, argmin = dists:min(1)
         min = min[1]
         argmin = argmin[1]
      
         -- average template:
         kernels[argmin] = (kernels[argmin]*kernels_avg[argmin] + sample)/(kernels_avg[argmin]+1)
      
         -- update counter:
         kernels_avg[argmin] = kernels_avg[argmin] + 1
      
      end
      
      -- callback?
      if callback then callback(kernels) end
      
      -- progress
      xlua.progress(t, niter)      
   end  
   return kernels, kernels_avg
end


--function unsup.kmeans(x, k, std, niter, batchsize, callback, verbose)
--   -- args
--   batchsize = batchsize or 1000
--   std = std or 0.1
--
--   -- some shortcuts
--    sum = torch.sum
--    max = torch.max
--    pow = torch.pow
--    randn = torch.randn
--    zeros = torch.zeros
--
--   -- dims
--    nsamples = (#x)[1]
--    ndims = (#x)[2]
--
--   -- initialize means
--    x2 = sum(pow(x,2),2)
--    centroids = randn(k,ndims)*std
--    totalcounts = zeros(k)
--
--   -- do niter iterations
--   for i = 1,niter do
--      -- progress
--      if verbose then xlua.progress(i,niter) end
--
--      -- sums of squares
--       c2 = sum(pow(centroids,2),2)*0.5
--
--      -- init some variables
--       summation = zeros(k,ndims)
--       counts = zeros(k)
--       loss = 0
--
--      -- process batch
--      for i = 1,nsamples,batchsize do
--         -- indices
--          lasti = math.min(i+batchsize-1,nsamples)
--          m = lasti - i + 1
--
--         -- k-means step, on minibatch
--          batch = x[{ {i,lasti},{} }]
--          batch_t = batch:t()
--          tmp = centroids * batch_t
--         for n = 1,(#batch)[1] do
--            tmp[{ {},n }]:add(-1,c2)
--         end
--          val,labels = max(tmp,1)
--         loss = loss + sum(x2[{ {i,lasti} }]*0.5 - val:t())
--
--         -- count examplars per template
--          S = zeros(m,k)
--         for i = 1,(#labels)[2] do
--            S[i][labels[1][i]] = 1
--         end
--         summation:add( S:t() * batch )
--         counts:add( sum(S,1) )
--      end
--
--      -- normalize
--      for i = 1,k do
--         if counts[i] ~= 0 then
--            centroids[i] = summation[i]:div(counts[i])
--         end
--      end
--
--      -- total counts
--      totalcounts:add(counts)
--
--      -- callback?
--      if callback then callback(centroids) end
--   end
--
--   -- done
--   return centroids,totalcounts
--end
