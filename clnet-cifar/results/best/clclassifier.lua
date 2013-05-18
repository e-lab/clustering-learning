-- trains a CL lassifier based on clustering of train data:
function trainCLClassifier(fracDataSet,nclusters) -- param = fraction of dataset [0 to 1]
	-- split dataset into classes and train clusters for each category:
	local limitData = torch.ceil(trainData:size()*fracDataSet)
	if limitData%2 ~= 0 then limitData = limitData-1 end
	local splitdata = torch.Tensor(#classes, limitData/2, trainData.data:size(2))
	for i = 1,limitData do
		splitdata[trainData.labels[i]][torch.ceil(i/2)] = trainData.data[i]
		xlua.progress(i, limitData)
	end
	-- now run kmeans on each class:
	local clusteredClasses = torch.Tensor(#classes, nclusters, trainData.data:size(2))
	for i = 1,#classes do
		clusteredClasses[i] = okmeans(splitdata[i], nclusters, nil, 
				opt.initstd, opt.niter, opt.kmbatchsize, nil, verbose)
	end
	
	return clusteredClasses
end


function testCLnet(fracDataSet, clusteredClasses, nclusters)
	local limitDataTr = torch.ceil(trainData:size()*fracDataSet)
	if limitDataTr%2 ~= 0 then limitDataTr = limitDataTr-1 end
 	-- test on trainData: 
	local dist = torch.Tensor(#classes, nclusters)
	local correctTr = 0
	for i = 1,limitDataTr do
		local temp = trainData.data[i]
		--temp = temp - temp:mean() -- remove mean from input data
		--temp = temp / temp:std()
		for j=1,#classes do
			for k=1,nclusters do
				--dist[j][k] = SMRmatch(temp:reshape((#temp)[1]), clusteredclasses[j][k], 0.75)
				dist[j][k] = torch.dist(temp, clusteredClasses[j][k])
			end
		end
		max, idx = torch.min(torch.min(dist,2),1)
		--print(idx[1][1])
		if ( trainData.labels[i] == idx[1][1] ) then 
			correctTr = correctTr+1 
		end
		--xlua.progress(i, limitDataTr)
	end
	print('Final correct percentage on trainData: '.. correctTr/limitDataTr*100)
	
	local limitDataTe = torch.ceil(testData:size()*fracDataSet)
	if limitDataTe%2 ~= 0 then limitDataTe = limitDataTe-1 end
	-- test on testData: 
	local correctTe = 0
	for i = 1,limitDataTe do
		local temp = testData.data[i]
		--temp = temp - temp:mean() -- remove mean from input data
		--temp = temp / temp:std()
		for j=1,#classes do
			for k=1,nclusters do
				dist[j][k] = torch.dist(temp, clusteredClasses[j][k])
			end
		end
		max, idx = torch.min(torch.min(dist,2),1)
		--print(idx[1][1])
		if ( testData.labels[i] == idx[1][1] ) then 
			correctTe = correctTe+1 
		end
		--xlua.progress(i, limitDataTe)
	end
	
	print('Final correct percentage on testData: '.. correctTe/limitDataTe*100)
	
	return correctTr/limitDataTr*100,correctTe/limitDataTe*100
end


-- this function does not work great...
function SMRmatch(in1, in2, ratio) -- only compares top ratio of highest values -- in2=template!
	local sin2, idxin2 = torch.sort(in2,true) -- we suppose in2 is the template (averaged sample)
	local indextokeep = torch.ceil(ratio*(#in1)[1])
	local distance = 0
	for i=1,indextokeep do
		distance = distance + torch.abs( in1[idxin2[i]] - in2[idxin2[i]] )
	end
	return distance
end

