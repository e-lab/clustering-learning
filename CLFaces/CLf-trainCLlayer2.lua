

-- 2nd layer:

require 'unsup'

print '==> extracting patches'
rpnum = 0
data = torch.Tensor(opt.nsamples,is*is)
i = 1
while i <= opt.nsamples do
   img = math.random(1,trainData2.data:size(1))
   img2 = trainData2.data[img] -- trainData1
   z = math.random(1,nk1)
   x = math.random(1,14-is+1)
   y = math.random(1,14-is+1)
   randompatch = img2[{ {z},{y,y+is-1},{x,x+is-1} }]
   if torch.sum(randompatch-randompatch) ~= 0 or randompatch:std() == 0 then 
      rpnum=rpnum+1
      print('Rejected randompatch!') 
   else
      data[i] = randompatch
      -- normalize patches to 0 mean and 1 std:
      data[i]:add(-randompatch:mean())
      data[i]:div(randompatch:std())
      i = i + 1
   end
end
print('Rejected pathes: ', rpnum)

-- show a few patches:
if opt.visualize then
   f256S = data[{{1,256}}]:reshape(256,is,is)
   image.display{image=f256S, nrow=16, padding=2, zoom=2, legend='Patches for 2nd layer learning'}
end

if not paths.filep('cifar10-1l.t7') then
   print '==> running k-means'
   function cb (kernels2)
      if opt.visualize then
         win2 = image.display{image=kernels2:reshape(nk,is,is), padding=2, symmetric=true, 
         zoom=2, win=win2, nrow=math.floor(math.sqrt(nk)), legend='2nd layer filters'}
      end
   end                    
   kernels2 = unsup.kmeans(data, nk, opt.initstd,opt.niter, opt.batchsize,cb,true)		
   print('==> saving centroids to disk:')
   --torch.save('faces-1l.t7', kernels)
end

print '==> verify filters statistics'
print('filters max mean: ' .. kernels2:mean(2):abs():max())
print('filters max standard deviation: ' .. kernels2:std(2):abs():max())
