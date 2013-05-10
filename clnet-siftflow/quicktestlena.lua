-- quitest lena code for nets:


normkernel = image.gaussian1D(9)
normer=nn.SpatialContrastiveNormalization(3, normkernel,1e-3)
lvn=normer:forward(image.lena())
lv1 = vnet:forward(lvn)
image.display(lv1)
lv2 = vnet2:forward(lv1)
image.display(lv2)



vnet = nn.Sequential()
vnet:add(nn.SpatialConvolution(1, nk1, is, is, cvstepsize,cvstepsize))
--vnet:add(nn.Tanh())
vnet:add(nn.HardShrink(0.1))
--vnet:add(nn.Abs())
vnet:add(nn.SpatialMaxPooling(s0_,s0_,s0_,s0_))
--vnet:add(nn.SpatialContrastiveNormalization(nk1, normkernel,1e-3))

vnet.modules[1].weight = kernels1_:reshape(nk1, 1, is,is):expand(nk1,ivch,is,is)/14

lv1 = vnet:forward(lvn)
image.display(lv1)
return vnet.modules[1].output:std()
return vnet.modules[1].output:mean()
return lv1:std()
return lv1:mean()