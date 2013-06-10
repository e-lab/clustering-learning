-- Plotter of test/train data
-- E. Culurciello
-- August 2nd 2012

torch.setdefaulttensortype('torch.DoubleTensor')

file=io.open('results/train.log', 'r')
trd = {}
i=0
while 1 do
    i = i+1
    local l = file:read() -- first line skip
    l = file:read() -- read data
    if not l then i=0 break end
    trd[i] = l
end
trd=torch.Tensor(trd)
file:close()

file=io.open('results/test.log', 'r')
ted = {}
i=0
while 1 do
    i = i+1
    local l = file:read() -- first line skip
    l = file:read() -- read data
    if not l then i=0 break end
    ted[i] = l
end
ted=torch.Tensor(ted)
file:close()

xtrd = torch.linspace(1, trd:size(1), trd:size(1))
xted = torch.linspace(1, ted:size(1), ted:size(1))
--gnuplot.closeall()
--gnuplot.figure(1)
--gnuplot.title('Network Results')
gnuplot.epsfigure('results.eps')
gnuplot.plot({xtrd, trd, '-'}, {xted, ted, '-'})
gnuplot.xlabel('epoch #')
gnuplot.ylabel('Precision %')
gnuplot.movelegend('right','bottom')
gnuplot.raw('set terminal postscript enhanced "Helvetica" 16')
gnuplot.plotflush()

gnuplot.epsfigure('resultsb.eps')
gnuplot.plot({xtrd, trd, '-'}, {xted, ted, '-'})
gnuplot.xlabel('epoch #')
gnuplot.ylabel('Precision %')
gnuplot.movelegend('right','bottom')
gnuplot.raw('set terminal postscript enhanced "Helvetica" 24')
gnuplot.plotflush()