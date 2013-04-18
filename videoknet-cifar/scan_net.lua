-- ECulurciello, March 2013: study of onlin-learner net
-- to figure out how layers connect

require 'nnx'
require 'image'
require 'trainLayer'

net = torch.load('../online-learner/encoder.net'):double()

vdata_in = image.scale(image.lena(),128,128)
vdata = net:forward(vdata_in)
vdata = net.modules[1].output
--remove mean and normalize std, to compare features correctly:
for i =1,16 do
   vdata[i]:add(-vdata[i]:mean()):div(vdata[i]:std())
end
image.display{image=vdata, symmetric=true, nrow=8}
vdata = net.modules[1].output:resize(1,16,122,122)

-- vars:
nlayer=2
nnf2=1
nsamples = 10000
is = net.modules[4].kW
nf = 1
mode = 'AND'
fanin = 2
nk1 = net.modules[4].nInputPlane
nkp = nk1
nk2 = net.modules[4].nOutputPlane
nkn = nk2
cTable = net.modules[4].connTable
weights = net.modules[4].weight

-- print some filters:
image.display{image=net.modules[1].weight, padding=2, symmetric=true, nrow=8, zoom=4}
image.display{image=net.modules[4].weight[{{1,32}}], padding=2, symmetric=true,nrow=8, zoom=4}
      
-- compute covariance matrix:
covMat = torch.zeros(nk1,nk1) --covariance matrix
for k=1,nf do
   for i=1,nkp do
      for j=i,nkp do
         if mode == 'OR' then covMat[i][j] = covMat[i][j] + torch.dist(vdata[k][i]:abs(), vdata[k][j]:abs()) -- dist metric
         elseif mode == 'AND' then covMat[i][j] = covMat[i][j] + torch.cmul(vdata[k][i]:abs(), vdata[k][j]:abs()):abs():sum() -- conv metric
         else print('Error: mode must be AND or OR!') end
         covMat[j][i] = covMat[i][j]
      end
   end
end  

-- sort connections and show them:
sortedMat = torch.zeros(nk1,nk1)
for i=1,nkp do
   if mode == 'OR' then max, inx = torch.sort(covMat[i]) end --want smaller values first (dist)
   if mode == 'AND' then max, inx = torch.sort(covMat[i], true) end -- want larger values first (conv)
   -- create sorted matrix
   sortedMat[i] = inx 
end

print(sortedMat)
  


-- create connex matrix to visualize:
-- row = index of maps in next layer, colums = maps group from previous layer
connMat = torch.zeros(nk2,8)
j = 1
for i=1, cTable:size(1) do -- because net fanin is 8 here
   connMat[torch.ceil(i/8)][j] = cTable[i][1]
   if j==8 then j=1 else j = j+1 end
end
--print(connMat)

-- create connex histogram:
connHist = torch.zeros(nkp)
for i=1, cTable:size(1) do -- because net fanin is 8 here
   connHist[cTable[i][1]] = connHist[cTable[i][1]] + 1
end
--print(connHist)



--print(cTable[{{1,32}}])
--
--for i=1,4 do
--   dispdata = net.modules[1].output:clone()
--   for j=1,8 do
--      dispdata[cTable[(i-1)*8+j][1]] = dispdata[cTable[(i-1)*8+j][1]] * 0
--   end
--   image.display{image=dispdata, zoom = 0.25}
--end



-- there seems to be some similarity between the prev layer filter group (set by connex matrix) and the next layer filters
-- are the next filters averages of the prev layer filter group (set by connex matrix)?