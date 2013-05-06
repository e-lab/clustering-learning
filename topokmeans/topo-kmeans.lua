
-- Topographic k-means onld stye: no batch - May 4th 2013
-- supposes a k = pow(2)

-- E. Culurciello
-- The k-means algorithm. ONLINE VERSION by E. Culurciello Jan 2013
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
function topokmeans(x, k, centroids, std, niter, batchsize, callback, verbose)
   -- args
   --batchsize = batchsize or 1000
   std = std or 0.1 -- should be:  x:std(2):std()

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]

   -- initialize means
   if not(centroids) then
      centroids = torch.randn(k,ndims)*std
   end
   local totalcounts = torch.zeros(k)

   -- do niter iterations
   for i = 1,niter do
      -- progress
      if verbose then xlua.progress(i,niter) end

      -- init some variables
      --local summation = torch.zeros(k,ndims)
      local counts = torch.ones(k)

      -- process
      for i = 1,nsamples do
      	local minjdx = 0
      	local centDist = torch.Tensor(k)
      	for j = 1,k do -- find closest matches with centroids:
      		centDist[j] =  torch.dist( centroids[j], x[i] )
      	end
      	local dsorted 
      	_, dsorted = torch.sort(centDist) 
      	
      	local centToAve = 1 -- number of centroids to average with current sample 
      	for j = 1, centToAve do
      		local a = dsorted[j]
      		centroids[a] = (centroids[a]*counts[a] + x[i]) / ( counts[a] + 1)
      		counts[a] = counts[a] + 1
      		
      		-- also average topographically to neighbors:
      		local topo = false
      		if topo  then
					local sqk = torch.sqrt(k)

					-- h+
					local b = torch.floor((a-1)/sqk)*sqk + ((a-1)+1)%sqk+1
					centroids[b] = (centroids[b]*counts[b] + x[i]) / ( counts[b] + 1)
					counts[b] = (counts[b] + 1)
					-- h-
					local c = torch.floor((a-1)/sqk)*sqk + ((a-1)-1)%sqk+1
					centroids[c] = (centroids[c]*counts[c] + x[i]) / ( counts[c] + 1)
					counts[c] = counts[c] + 1
					-- v+
					local d = ((a-1)+sqk)%k+1
					centroids[d] = (centroids[d]*counts[d] + x[i]) / ( counts[d] + 1)
					counts[d] = counts[d] + 1
					-- v-
					local e = ((a-1)-sqk)%k+1
					centroids[e] = (centroids[e]*counts[e] + x[i]) / ( counts[e] + 1)
					counts[e] = counts[e] + 1     		
      		
      			--print(a,b,c,d,e)
      		end
      	end
      	--callback(centroids)
--      	require 'sys'
--      	sys.sleep(0.5)
      end
      
--1	2	3	4	5	6	7	8
--9	10	11	12	13	14	15	16
--17	18	19	20	21	22	23	24
--25	26	27	28	29	30	31	32
--33	34	35	36	37	38	39	40
--41	42	43	44	45	46	47	48
--49	50	51	52	53	54	55	56
--57	58	59	60	61	62	63	64
      
      -- normalize
      for i = 1,k do
         --if counts[i] ~= 0 then
            --centroids[i] = summation[i]:div(counts[i])
         --end
         centroids[i] = centroids[i] - centroids[i]:mean()
         centroids[i] = centroids[i]/centroids[i]:std()
      end

      -- total counts
      totalcounts:add(counts)

      -- callback?
      if callback then callback(centroids) end
   end

   -- done
   return centroids,totalcounts
end
