-- Plotter of test/train data for CL paper
-- E. Culurciello
-- October 2012

-- HELP:
-- http://www2.yukawa.kyoto-u.ac.jp/~ohnishi/Lib/gnuplot.html


function read_data_file(file_name)
   local file=io.open(file_name, 'r')
   local data = {}
   local i=0
   while 1 do
       i = i+1
       local l = file:read() -- first line skip
       l = file:read() -- read data
       if not l then i=0 break end
       data[i] = l
   end
   file:close()
   return torch.Tensor(data)
end



--trd = read_data_file('results/cifar-CL1l-dist-16-filters/train.log')
ted1 = read_data_file('results/cifar-CL1l-dist-16-filters/test.log')
xted1 = torch.linspace(1, ted1:size(1), ted1:size(1))
ted2 = read_data_file('results/cifar-CL2l-dist-16-128/test.log')
xted2 = torch.linspace(1, ted2:size(1), ted2:size(1))
ted3 = read_data_file('results/cifar-convnet-1L-16-128/test.log')
xted3 = torch.linspace(1, ted3:size(1), ted3:size(1))
ted4 = read_data_file('results/cifar-convnet-16-128/test.log')
xted4 = torch.linspace(1, ted4:size(1), ted4:size(1))
--gnuplot.closeall()
--gnuplot.figure(1)
--gnuplot.title('Network Results')
gnuplot.epsfigure('results.eps')
gnuplot.raw('set linestyle  1 lt 1 lc 7 # black-solid')
gnuplot.plot({'CL-1l', xted1, ted1, '+-'}, {'CL-2l', xted2, ted2, '-'}, {'convnet-1l', xted3, ted3, '+-'}, {'convnet-12l', xted4, ted4, '-'})
gnuplot.xlabel('epoch #')
gnuplot.ylabel('Precision %')
gnuplot.movelegend('right','bottom')
gnuplot.raw('set terminal postscript enhanced "Helvetica" 24')
gnuplot.plotflush()