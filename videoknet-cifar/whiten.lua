-- E. Culurciello, March 2013
-- whitening ZCA code from Adam Coates, translated from Matlab

-- original code from Jose Carrasco, Nov 2012

-- original matlab code:
--% whiten
--if (whitening)
--  C = cov(patches);
--  M = mean(patches);
--  [V,D] = eig(C);
--
--  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
--tengo que hacer:
----  P = V * diag(sqrt(1./(diag(e) + 0.1))) * V';
--  patches = bsxfun(@minus, patches, M) * P;
--end

require 'image'


data = image.scale(image.lena(),512,512)

a=data:clone()
outa = torch.Tensor(a)

-- normalize:
a:add(-a:mean())
a:div(a:std())


for i=1,a:size(1) do
   -- compute covariance matrix:
   cov = a[i]*a[i]:t()
   
   e,v = torch.eig(cov,'V')
   
   D = torch.diag(torch.diag(e:select(2,1))+0.1)
   
   on2 = torch.ones(D:size(1))
   on2:diag(on2:sqrt(on2:cdiv(D)))

   P = v*on2*(v:transpose(1,2))

   a[i] = a[i]*P
end


image.display(a)


