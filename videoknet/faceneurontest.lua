----------------------------------------------------------------------
-- now MAGIC: find neurons that might encode for a face and NOT bg:

-- image.display(trainData[n][1]) -- to see if face or not
-- if trainData[n][2][1] == 1 --it is a face (if bg returns -1)

-- function max
function GetMax(a)
	x,xi = torch.max(a,1)
	y,yi = torch.max(x,2) -- y = value

	x_out = yi[1][1]  -- y coord
	y_out = xi[1][x_out] -- x coord
	return y,x_out,y_out 
end

-- max and indices for 3D tensors:
function GetMax3D(a)
	m1,i1 = torch.max(a,1)
	m2,i2 = torch.max(m1,2)
	m3,i3 = torch.max(m2,3)

	i3_out = i3[1][1][1]
	i2_out = i2[1][1][i3_out]
	i1_out = i1[1][i2_out][i3_out]
	return m3,i1_out,i2_out,i3_out
end

-- min and indices for 3D tensors:
function GetMin3D(a)
	m1,i1 = torch.min(a,1)
	m2,i2 = torch.min(m1,2)
	m3,i3 = torch.min(m2,3)

	i3_out = i3[1][1][1]
	i2_out = i2[1][1][i3_out]
	i1_out = i1[1][i2_out][i3_out]
	return m3,i1_out,i2_out,i3_out
end


-- let's average all face output and look for most active neurons:
facefound = 0
faceave = torch.zeros(#trainData2.data[1])
for i = 1, trainsize do
   if (trainData[i][2][1] == 1) then
      facefound = facefound + 1
      faceave = faceave + trainData2.data[i]:abs() -- abs to reflect best in both directions
   end
end
faceavenorm = faceave/facefound
print('Number of faces found in train set:', facefound)
image.display{image=faceavenorm, padding=2, zoom=4}


-- average of input image for top face neuron:
o_size = trainData2.data[1]:size(2)
vmax, i1, i2, i3 =  GetMax3D(faceavenorm) -- max neuron based on average of all faces
tnave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   tnave = tnave + trainData[i][1]*trainData2.data[i][i1][i2][i3]:abs() -- average dataset weighted by top neuron
end
tnave:resize(1,32,32):div(trainsize)
image.display{image=tnave, padding=2, zoom=4, legend='average dataset weighted by top neuron'}


-- average of input image for worse face neuron:
o_size = trainData2.data[1]:size(2)
vmin, i1, i2, i3 =  GetMin3D(faceavenorm) -- min neuron based on average of all faces
tnave2 = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   tnave2 = tnave2 + trainData[i][1]*trainData2.data[i][i1][i2][i3]:abs() -- average dataset weighted by worse neuron
end
tnave2:resize(1,32,32):div(trainsize)
image.display{image=tnave2, padding=2, zoom=4, legend='average dataset weighted by worse neuron'}


-- countertest: random neuron:
rndave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   rndave = rndave + trainData[i][1]*trainData2.data[i][35][1][2] -- average dataset weighted by random neuron 1,1
end
image.display{image=rndave/trainsize, padding=2, zoom=4, legend='average dataset weighted by random neuron'}


-- average of dataset:
dsave = torch.zeros(#trainData[1][1])
for i = 1, trainsize do
   dsave = dsave + trainData[i][1] -- average of all dataset
end
image.display{image=dsave/trainsize, padding=2, zoom=4, legend='average dataset'}