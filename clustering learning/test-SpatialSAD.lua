----------------------------------------------------------------------
-- Run k-means on CIFAR10 dataset - 1st layer test
----------------------------------------------------------------------

import 'torch'
require 'nnx'
require 'image'
require 'unsup'
require 'SpatialSAD'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Get k-means templates on directory of images')
cmd:text()
cmd:text('Options')
cmd:option('-images', 'images', 'directory full of images')
cmd:option('-maximages', 100, 'max nb of images')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 8, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches') -- 9x9 kernels wanted
cmd:option('-nkernels', 64, 'number of kernels to learn')
cmd:option('-niter', 50, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 1000*100, 'nb of random training samples')
cmd:option('-initstd', 0.1, 'standard deviation to generate random initial templates')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
params = cmd:parse(arg or {})

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)
torch.setdefaulttensortype('torch.DoubleTensor')

is = params.inputsize

----------------------------------------------------------------------
print '==> doing small test for checking internals:'

iis = 2
ni = 3
no = 4
kernels = torch.Tensor({{1,0,0,1},{0,1,1,0},{1,1,1,1},{0,0,0,0}})
lena = torch.Tensor({{{1,1},{1,1}},{{1,0},{0,1}},{{0,1},{0,1}}})

n1 = nn.Sequential()
n1:add(nn.SpatialSAD(ni,no,iis,iis))
n1.modules[1]:templates(kernels:reshape(no,1,iis,iis):expand(no,ni,iis,iis))

time = sys.clock()
outima = n1:forward(lena)
time = sys.clock() - time
print('==> Compute Time = ' .. (time*1000) .. 'ms')
print(lena, kernels, outima)
-- test passed!
--n1:templates(time) -- my break function!


----------------------------------------------------------------------
print '==> loading test kernels'
kernels = torch.load('cifar10-1l-64.t7')
lena = image.scale(image.lena(), 128, 128)
nk = kernels:size(1)
is = 9

--print '==> computing distances to templates, convolutionally SAD2 faster version'
--require 'SpatialSAD2'
--n2 = nn.Sequential()
--n2:add(nn.SpatialSAD2(1,nk,is,is))
--n2.modules[1]:templates(kernels:reshape(nk, 1, is, is))
--
--time = sys.clock()
--outiman = n2:forward(lena)
--time = sys.clock() - time
--print('==> Compute Time = ' .. (time*1000) .. 'ms')
--image.display(outiman[1])


print '==> computing distances to templates SAD:'
n1 = nn.Sequential()
n1:add(nn.SpatialSAD(3,nk,is,is))
n1.modules[1]:templates(kernels:reshape(nk, 1, is, is):expand(nk,3,is,is))
n1:add(nn.Sum(2))

time = sys.clock()
outima = n1:forward(lena)
time = sys.clock() - time
print('==> Compute Time = ' .. (time*1000) .. 'ms')
--image.display(outima[1])

-- comparing speed with spatial convolution layer:
print '==> computing spatial convolution on same image:'
n2 = nn.Sequential()
n2:add(nn.SpatialConvolution(3, nk, is, is, cvstepsize, cvstepsize))
n2:add(nn.Sum(2))
-- initialize 1st layer parameters to learned filters:
for i=1,nk do   
   n2.modules[1].weight[i] = kernels[i]:reshape(3, is, is):type('torch.FloatTensor')
end
n2.modules[1].bias = n2.modules[1].bias *0

time = sys.clock()
outima = n2:forward(lena)
time = sys.clock() - time
print('==> Compute Time = ' .. (time*1000) .. 'ms')
--image.display(outima[1])



------------------------------------------------------------------------
--print '==> testing with one layer neural net:'
normkernel = image.gaussian1D(7)
o1size = lena:size(3) - is + 1 -- size of spatial conv layer output
--cvstepsize = 2
poolsize = 2
--outsize = 10

l1net = nn.Sequential()
l1net:add(nn.SpatialSAD(3,nk,is,is))
l1net.modules[1]:templates(kernels:reshape(nk, 1, is, is):expand(nk,3,is,is))
l1net:add(nn.Sum(2))
l1net:add(nn.SpatialContrastiveNormalization(nk, normkernel, 1e-3))
l1net:add(nn.Tanh())
l1net:add(nn.SpatialLPPooling(nk,2,poolsize,poolsize,poolsize,poolsize))
l1net:add(nn.SpatialSubtractiveNormalization(nk, normkernel))

time = sys.clock()
outima2 = l1net:forward(lena)
time = sys.clock() - time
print('==> Compute Time = ' .. (time*1000) .. 'ms')
image.display(outima2[1])





