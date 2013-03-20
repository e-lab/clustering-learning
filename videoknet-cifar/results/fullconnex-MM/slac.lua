-- single link agglomerative clustering
function slac(feats,startN,finalN,tau,Delta,Lp)
   tau = tau or 0.3 -- max distance to add to cluster
   Delta = Delta or 1.5 -- maximum diameter
   Lp = Lp or 2 -- use Lp distance for clustering

   local featN = feats:size(1)
   --local diameters = torch.Tensor(startN):zero()
   local shufl1 = torch.randperm(featN)
   local clust = {}
   local clustSize = torch.Tensor(startN):fill(1)
   for i = 1,startN do
      local group = {}
      group[shufl1[i]] = feats[shufl1[i]]
      clust[i] = group
   end

   -- this loop will be parallelized when rewritten in C
   for i = 1,startN do
      local group = clust[i]
      local seed = shufl1[i]
      local shufl2 = torch.randperm(featN)
      for j = 1,featN do
         local idx_j = shufl2[j]
         if idx_j ~= seed then
            local ft_j = feats[idx_j]
            local minDist = math.huge
            local maxDist = 0
            for idx_k,ft_k in pairs(group) do
               dist = torch.dist(ft_j,ft_k,Lp)
               if dist > maxDist then
                  maxDist = dist
               end
               if maxDist > Delta then
                  break
               end
               if dist < minDist then
                  minDist = dist
               end
            end
            if minDist < tau then
               group[idx_j] = ft_j
               clustSize[i] = clustSize[i] + 1
               if maxDist >= Delta then
                  break
               end
            end
         end
      end
   end
   --print('clust:')
   --for i,grp in pairs(clust) do
   --   print(grp)
   --end

   -- sort clusters by size 
   local sSz,sId = torch.sort(clustSize,1,true)

   -- retain finalN clusters
   local function containsGroup(clusters,group)
      for gidx,grp in pairs(clusters) do
         if table.getn(grp) == table.getn(group) then
            local geq = true
            for idx,ft in pairs(group) do
               if grp[idx] == nil then
                  geq = false
                  break
               end
            end
            if geq then
               return true
            end
         end
      end
      return false
   end 
   local function disjointSet(clusters,group)
      for gidx,grp in pairs(clusters) do
         if table.getn(grp) == table.getn(group) then
            local geq = true
            for idx,ft in pairs(group) do
               if grp[idx] == nil then
                  return false
               end
            end
         end
      end
      return true
   end 
   local finalClust = {}
   local featsRetained = {}
   local finalFeats = feats:clone()
   local totalFeatsRetained = 0
   local totalConnections = 0
   local finalClustN = 0
   local i = 1
   while finalClustN < finalN do
      local idx_cl = sId[i]
      local group = clust[idx_cl]
      local finalGroup = {}
      local new_feats = true
      for idx_j, ft_j in pairs(group) do
         local f_idx = featsRetained[idx_j]
         if f_idx ~= nil then
            new_feats = false
         end
      end
      if new_feats then
         local group_sz = 0
         for idx_j, ft_j in pairs(group) do
            totalFeatsRetained = totalFeatsRetained + 1
            local final_idx = totalFeatsRetained
            featsRetained[idx_j] = final_idx
            finalFeats[final_idx] = ft_j
            finalGroup[final_idx] = ft_j
            group_sz = group_sz + 1
         end
         finalClustN = finalClustN + 1
         finalClust[finalClustN] = finalGroup
         totalConnections = totalConnections + group_sz
      end
      i = i + 1
      if i > startN then
         break
      end
   end
   assert(finalClustN == finalN, 'not enough distinct clusters, try larger startN or smaller finalN')
   finalFeats = finalFeats:sub(1,totalFeatsRetained)
   --print('totalFeatsRetained',totalFeatsRetained)
   --print('finalClust:')
   --for i,grp in pairs(finalClust) do
   --   print(grp)
   --end
   --print('totalConnections',totalConnections)

   -- construct connection table
   local connTable = torch.Tensor(totalConnections,2):zero()
   local c_j = 1
   for i,group in pairs(finalClust) do
      for idx_j,ft_j in pairs(group) do
         connTable[c_j][1] = idx_j
         connTable[c_j][2] = i
         c_j = c_j + 1
      end
   end
   --print('c_j',c_j)
   assert(c_j == totalConnections + 1, 'error constructing connection table')
   
   return finalFeats, connTable
end

function getGroups(connTable,feats)
   local groups = {}
   for i = 1,connTable:size(1) do
      feat = feats[connTable[i][1]]
      group_num = connTable[i][2]
      if groups[group_num] == nil then
         groups[group_num] = {}
      end
      local group = groups[group_num]
      group[#group + 1] = feat
   end
   return groups
end

function showGroup(group,kW,kH,zoom,padding)
   local kW = kW or 5
   local kH = kH or 5
   local zoom = zoom or 8
   local padding = padding or 2
   for i,img in pairs(group) do
      img:resize(5,5)
   end
   image.display{image=group,zoom=zoom,padding=padding}
end

function showGroups(groups)
   for i,group in pairs(groups) do
      showGroup(group)
   end
end



-- test EC for output connex:
--maps, conx = slac(videoData2[1], nk1, nk1/2, 100, 70)
--gp= getGroups(conx,maps)
--showGroup(gp[1],videoData2[1]:size(2),videoData2[1]:size(3),1,2) --only returns small kernels... why?

