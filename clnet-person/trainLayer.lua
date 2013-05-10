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
      if patches[{{1}}]:std() > patchminstd then
         patches:add(-patches:mean())
         patches:div(patches:std()+1e-3) -- to prevent divide-by-0
         -- shape by gaussian:
         data[i] = torch.cmul(patches, image.gaussian(is, 0.5):reshape(1,1,is,is):expand(ivch,nnf,is,is))
         i = i+1 -- if patches is used then count up
      end      
      if verbose then xlua.progress(i, nsamples) end
   end
   
   -- show a few patches:
--   if verbose then
--      f256S = data[{{1,256}}]:reshape(256,ivch,nnf*is,is)
--      image.display{image=f256S, nrow=16, nrow=16, padding=2, zoom=2, legend='Patches of input data before clustering'}
--   end
   
   ----------------------------------------------------------------------
   if verbose then print '==> running k-means to learn filter' end
   
   local win
   function cb (kernels)
      if verbose then
         win = image.display{image=kernels:reshape(nk,ivch, nnf*is, is), padding=2, symmetric=true, 
         zoom=4, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
      end
   end                    
   
   --local kernels = torch.Tensor(nk, ivch, nnf, is, is)
   --local counts = torch.Tensor(nk, ivch, 1)
   --kernels = kmec(data, nk, opt.initstd, opt.niter, opt.kmbatchsize, cb, true) -- Euge kmeans (not good init yet)
   --kernels = unsup.kmeans(data, nk, opt.initstd, opt.niter, opt.kmbatchsize, cb, true)
   kernels, counts = okmeans(data, nk, nil, opt.initstd, opt.niter, opt.kmbatchsize, cb, verbose)
   --kernels, counts = topokmeans(data, nk, nil, opt.initstd, opt.niter, nil, cb, verbose)
--   for i = 1, ivch do
--      kernelss, countss = okmeans(data[{{},{i}}]:reshape(nsamples,nnf*is*is), nk, nil, opt.initstd, opt.niter, opt.kmbatchsize, cb, verbose) -- online version to upadte filters
--      kernels[{{},{i}}] = kernelss:reshape(nk,1,nnf, is, is)
--      counts[{{},{i}}] = countss:reshape(nk,1,1)
--   end
   
   for i=1,nk do
      -- normalize kernels to 0 mean and 1 std:
      --kernels[i]:add(-kernels[i]:mean())
      --kernels[i]:div(kernels[i]:std())
   
      -- clear nan kernels   
      if torch.sum(kernels[i]-kernels[i]) ~= 0 then 
         print('Found NaN kernels!') 
         kernels[i] = torch.zeros(kernels[1]:size()) 
      end
   end
   -- print final filters:
   if verbose then
      win = image.display{image=kernels:reshape(nk, ivch, nnf*is, is), padding=2, symmetric=true, 
         zoom=4, win=win, nrow=math.floor(math.sqrt(nk)), legend='Layer '..nlayer..' filters'}
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
   
   return kernels, counts
end



function processLayer(lv, network, data_in, nkernels, oheight, owidth, verbose)
   local data_out = torch.Tensor(data_in:size(1), nkernels, oheight, owidth)
   local stdc1=0
   local meac1=0
   local stdo=0
   local meao=0
   for i = nnf1, data_in:size(1) do
      if ( nnf1>1 and lv == 1 ) then procFrames = data_in[{{i-nnf1+1,i},{},{}}]:transpose(1,2) -- swap order of indices here for VolConvolution to work
      else procFrames = data_in[i] end
      data_out[i] = network:forward(procFrames)
      --stats: 
      stdc1 = stdc1 + network.modules[1].output:std()
      meac1 = meac1 + network.modules[1].output:mean()
      stdo = stdo + network.output:std()
      meao = meao + network.output:mean()      
      
      xlua.progress(i, data_in:size(1))
      -- do a live display of the input video and output feature maps 
      if verbose then
         winm = image.display{image=data_out[i], padding=2, zoom=1, win=winm, nrow=math.floor(math.sqrt(nkernels))}
      end
   end
   -- data_out = nil --free memory if needed
   return data_out, stdc1/data_in:size(1), meac1/data_in:size(1), 
   						stdo/data_in:size(1), meao/data_in:size(1)
end



function createCoCnx(nlayer, vdata, nkp, fpgroup, fanin, samples, nnf, is, verbose)
   -- create a covariance/co-occurence connection table based on some test data
   -- input data has multiple planes, compute similarity between these planes
   -- group planes that are similar
   -- we connect each group to one neuron in next layer
   
   -- train filter for next layer based on groups of connTable!!!
   -- uses co-occurence of features on muliple maps: sum maps, run clustering on them

   -- nkp = #  features previous layer, fpgroup = features per group
   -- fanin = desired connex fanin - this should be also learned from data...   
   -- nnf = number frames, is = kernel size
   -- verbose = true ==> show images, text messages
   
   assert(nkp == vdata:size(2), 'Error: nkp and input video features are not the same number!') -- number features
   
   local nf = vdata:size(1) --number frames
   local vd1 = torch.zeros(vdata:size(1),fanin,vdata:size(3),vdata:size(4)) --temp data storage
   local vd2 = torch.zeros(vdata:size(1),2*fanin,vdata:size(3),vdata:size(4)) --temp data storage
   
   local covMat = torch.zeros(nkp,nkp) --covariance matrix
   local connTable = {}--torch.Tensor( fanin*(1+2)*fpgroup ,2) -- table of connections
   local kerTable = {} -- table of kernels/filters
      
   if verbose then print 'createCoCnx: compute covariance matrix:' end
   for k=1,nf do
      for i=1,nkp do
         for j=i,nkp do
         	--torch.cmul(vdata[k][i], vdata[k][j]):sum()  -- instead fo torch.dist one can use mult
            covMat[i][j] = covMat[i][j] + torch.dist(vdata[k][i], vdata[k][j]) -- dist metric
            covMat[j][i] = covMat[i][j] -- replicate on lower part of matrix, since symmetric
         end
      end
   end   
   
   if verbose then print 'createCoCnx: connect cells in fanin groups:' end
   for i=1,nkp do
      max, inx = torch.sort(covMat[i]) --want smaller values first (dist)
      
      -- groups of fanin (connect top two max):
      for j=1,fpgroup do -- repeat to have multiple filter kernels from this group:
         for k=1,fanin do -- the first value connects to itself!
            table.insert(connTable, torch.Tensor({inx[k],i}))
            -- group all feature maps that co-occur / are similar
            if j == 1 then vd1[{{},{k}}] = vdata[{{},{inx[k]}}] end
         end
      end

      kerp = trainLayer(nlayer, vd1, samples, nil, fpgroup, nnf, is, verbose)
      for j=1,fpgroup do
         for k=1,fanin do
            table.insert(kerTable, kerp:reshape(fpgroup, fanin,is*is)[j][k])
         end
      end
      
      -- groups of 2 x fanin: (offset in i: 2*nkp)
      for j=1,fpgroup do -- repeat to have multiple filter kernels from this group:
         for k=1,2*fanin do -- the first value connects to itself!
            table.insert(connTable, torch.Tensor({inx[k],i+nkp}))
            -- group all feature maps that co-occur / are similar
            if  j == 1 then vd2[{{},{k}}] = vdata[{{},{inx[k]}}] end
         end
      end
      kerp = trainLayer(nlayer, vd2, samples, nil, fpgroup, nnf, is, verbose)
      for j=1,fpgroup do
         for k=1,2*fanin do
            table.insert(kerTable, kerp:reshape(fpgroup, fanin*2,is*is)[j][k])
         end
      end
      
      if verbose then xlua.progress(i, nkp) end
   end
 
   if verbose then print 'createCoCnx: turn tables into tensors:' end
   local connTableTensor = torch.Tensor(#connTable,2)
   local kerTensor = torch.zeros(#kerTable,is,is)
   for i, value in ipairs(connTable) do
      connTableTensor[i] = value
   end
   for i, value in ipairs(kerTable) do
      kerTensor[i] = value
   end
   
   if verbose then print 'createCoCnx: renormalize all kernels:' end
   for i=1,kerTensor:size(1) do
      kerTensor[i] = kerTensor[i]:add(-kerTensor[i]:mean()):div(kerTensor[i]:std())
   end
   
   return connTableTensor, kerTensor
end


-- tests:
-- image.display{image=kernels1:reshape(32,7,7), padding=2, zoom=4,  nrow=8}
-- image.display{image=videoData2[6], padding=2, zoom=1,  nrow=8}


function colorBypass(cnpoolsize, trainDataIN , testDataIN)
	-- Color Bypass function to concatenate final network output 
	-- with a subsampled version of the input
	-- E. Culurciello, May 2013
	
	-- cnpoolsize = pooling amount of deep net
	
	print "==> Color bypass: creating final test dataset:"
	local trsize = trainData:size()
	local tesize = testData:size()

	-- color bypass: downsamples color info and pass it to final classifier:
	colornet = nn.Sequential()
	colornet:add(nn.SpatialDownSampling(cnpoolsize,cnpoolsize,cnpoolsize,cnpoolsize))
	cdatasize = 3*(torch.floor(ivhe/cnpoolsize))^2 -- size of the color data

	-- process dataset throught net:
	trainDataF = {
		data = torch.Tensor(trsize, (#trainData2.data[1])[1]+cdatasize,1,1),
		color = torch.Tensor(trsize, cdatasize),  -- ad bypass color info
		labels = trainData.labels:clone(),
		size = function() return trsize end
	}
	testDataF = {
		data = torch.Tensor(tesize, (#testData2.data[1])[1]+cdatasize,1,1),
		color = torch.Tensor(trsize, cdatasize),  -- ad bypass color info
		labels = testData.labels:clone(),
		size = function() return tesize end
	}

	print '==> Color bypass: process color info of dataset throught colornet:'
	for t = 1,trsize do
		trainDataF.color[t] = colornet:forward(trainData.data[t][{{1,3}}])
		xlua.progress(t, trainData:size())
	end
	for t = 1,tesize do
		testDataF.color[t] = colornet:forward(testData.data[t][{{1,3}}])
		xlua.progress(t, testData:size())
	end

	print '==> Color bypass: concatenating dataset into final vector:'
	for t = 1,trsize do
		trainDataF.data[t] = torch.cat(trainData2.data[t]:reshape((#trainData2.data[1])[1]), trainDataF.color[t]):reshape((#trainData2.data[1])[1]+cdatasize ,1,1)
		xlua.progress(t, trsize)
	end
	for t = 1,tesize do
		testDataF.data[t] = torch.cat(testData2.data[t]:reshape((#trainData2.data[1])[1]), testDataF.color[t]):reshape((#testData2.data[1])[1]+cdatasize ,1,1)
		xlua.progress(t, tesize)
	end

	return trainDataF, testDataF

end




