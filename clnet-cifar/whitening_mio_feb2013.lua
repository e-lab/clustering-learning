

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


a=data:clone()
a2=a:clone()
b=a:clone()

for i=1,a:size(1) do

b[i]=b[i]-b[i]:mean()
end


meanm = {}
stdm = {}
for i=1,a:size(1) do
   -- normalize each channel globally:
   meanm[i] = a[{i,{} }]:mean()
   stdm[i]=torch.sqrt(a[{i,{} }]:var()+10)
   a[{i,{} }]:add(-meanm[i])
   a[{i,{} }]:div(stdm[i])
end




--a=data:clone()
 --a=data[{{1,4},{1,6}}]
b=a:clone()
aorig=a:clone()
for i=1,a:size(2) do
   meanm[i] = b[{{},i }]:mean()
   b[{{},i }]:add(-meanm[i])
end

amean=a:mean()
cov=torch.zeros(a:size(2),a:size(2))
for i=1,a:size(2) do
for j=1,a:size(2) do

cov[i][j]= a:size(1)/(a:size(1)-1)*torch.cmul(b[{{},i }],b[{{},j }]):mean()


--falta la sqrt
end
end

e,v = torch.eig(cov,'V')

D=torch.diag(torch.diag(e:select(2,1))+0.1)

on2=torch.ones(D:size(1))
on2:diag(on2:sqrt(on2:cdiv(D)))

P=v*on2*(v:transpose(1,2))

P2=P:clone()
mean2 = {}
--patches = bsxfun(@minus, patches, M) * P;
for i=1,a:size(2) do
   mean2[i] = a[{{},i }]:mean()
   a[{{},i }]:add(-mean2[i])
end

a1=a:clone()
a=a*P
print('afinal2')
--print(a)
