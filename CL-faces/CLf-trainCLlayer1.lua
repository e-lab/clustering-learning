
require 'unsup'

print '==> extracting patches' 
rpnum = 0
data = torch.Tensor(opt.nsamples,is*is)
for i = 1,opt.nsamples do
   img = math.random(1,trainData:size())
   img2 = trainData[img][1] -- trainData1
   z = math.random(1,trainData[1][1]:size(1))
   x = math.random(1,trainData[1][1]:size(2)-is+1)
   y = math.random(1,trainData[1][1]:size(3)-is+1)
   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
   if torch.sum(randompatch-randompatch) ~= 0 then 
      rpnum=rpnum+1
      print(randompatch,img,z,x,y)
   else
      data[i] = randompatch
      -- normalize patches to 0 mean and 1 std:
      data[i]:add(-randompatch:mean())
      data[i]:div(randompatch:std())
   end
end
print('Found NaN randompatch, number:', rpnum)

-- show a few patches:
if opt.visualize then
   f256S = data[{{1,256}}]:reshape(256,is,is)
   image.display{image=f256S, nrow=16, padding=2, zoom=2, legend='Patches for 1st layer learning'}
end

if not paths.filep('cifar10-1l.t7') then
   print '==> running k-means'
   function cb (kernels)
      if opt.visualize then
         win = image.display{image=kernels:reshape(nk,is,is), padding=2, symmetric=true, 
         zoom=2, win=win, nrow=math.floor(math.sqrt(nk)), legend='1st layer filters'}
      end
   end                    
   kernels = unsup.kmeans(data, nk, opt.initstd,opt.niter, opt.batchsize,cb,true)		
   print('==> saving centroids to disk:')
   --torch.save('faces-1l.t7', kernels)
end

print '==> verify filters statistics'
print('filters max mean: ' .. kernels:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels:std(2):abs():max())