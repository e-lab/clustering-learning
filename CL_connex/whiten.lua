
----------------------------------------------------------------------
-- Author : Jonghoon Jin, Aysegul Dundar
-- Option for whitening
-- Whitening is only available for the first layer with this code
----------------------------------------------------------------------


require 'unsup'

function zca_whiten(x)          -- zca whitening is used
      local dims = x:size()        -- outputs are whitened data(x),
      local nsamples = dims[1]     -- M, P statistic (matrix)
      local ndims    = dims[2]
      local M = torch.mean(x, 1)
      local D, V = unsup.pcacov(x)
      x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
      local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
      local P = V * diag * V:t()
      x = x * P
      return x, M, P
   end


function whiten_image(data, M, P, is)   -- operation:  data = (data - M) x P
 
       -- from 2D image to a list of 1D patches
   local npatches = (32-is)+1
   local x = torch.zeros(npatches*npatches, 3*is*is)
   local m = 1
   for j = 1, npatches-1 do
      for k = 1, npatches-1 do
          -- slice 32x32 image (ignore 1px border) into 36 patches of 5x5
          x[{m}] = data[{{1},{},{j,j+is-1},{k,k+is-1}}]:reshape(3*is*is)   --{((i-1)*6+(j-1))*6+k}
          m = m+1
      end
   end
   

       -- do whitening
   x:add(torch.ger(torch.ones(npatches*npatches), M:squeeze()):mul(-1))
   x = x * P
   data2 = torch.zeros(3, npatches*is, npatches*is)                                                                     
   local m =1                                                                                                                   
    -- from the list of 1D patches to 2D image                                                                                   
   for j = 1, npatches-1 do                                                                                                  
      for k = 1, npatches-1 do                                                                                               
         local xi = 1 + (k-1)*is                                                                                            
         local yi = 1 + (j-1)*is
         data2[{{}, {yi,yi+is-1},{xi,xi+is-1}}] = x[{m}]:reshape(3,is,is)                                                 
         m = m+1                                                                                                             
      end                                                                                                                    
   end                                                                                                                       
   return data2                                                                                                                 
end


function whitenprocessLayer(network, traindata, testdata, M, P, verbose)
   data = whiten_image(traindata[{{1}, {}, {}, {}}], M, P, is1)
   a = network:forward(data)
   local train_out = torch.Tensor(traindata:size(1), a:size(1), a:size(2), a:size(3))
   local test_out  = torch.Tensor(testdata:size(1), a:size(1), a:size(2), a:size(3))
   
   for i = 1, traindata:size(1) do
      data = whiten_image(traindata[{{i}, {}, {}, {}}], M, P, is1)   
      train_out[i] = network:forward(data)      
      xlua.progress(i, traindata:size(1))
   end
   for i = 1, testdata:size(1) do
      data = whiten_image(testdata[{{i}, {}, {}, {}}], M, P, is1)
      test_out[i] = network:forward(data)      
      xlua.progress(i, testdata:size(1))
   end
   -- Normalize the output

   for i=1, train_out:size(2) do
      -- normalize each channel globally:
       mean = train_out[{ {},i,{},{} }]:mean()
       std = train_out[{ {},i,{},{} }]:std()
       train_out[{ {},i,{},{} }]:add(-mean)
       train_out[{ {},i,{},{} }]:div(std)
       test_out[{ {},i,{},{} }]:add(-mean)
       test_out[{ {},i,{},{} }]:div(std)
    end
   return train_out, test_out
end

