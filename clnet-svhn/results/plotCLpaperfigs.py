
# Plotter of test/train data for CL paper
# E. Culurciello
# March 2013

import math, numpy, matplotlib
from matplotlib.pyplot import *

def load_data_file( filename ):
   lines=open(filename,"r").readlines()
   data=[]
   for l in lines:
      a=l.split()
      if a[0][0]!='%':
         data=data+[float(a[0])]      
   return data

# load all needed data: CIFAR
cy1 = load_data_file('CL_extra/test.log')

cy4 = load_data_file('convnet_extra/test.log')
cx4 = range(0,len(cy4))
cx1 = range(0,len(cy4))
figure(0)
plot(cx1, cy1, 'g-', cx4, cy4,'b--')
plot.ylim((40, 80))
xlabel('epoch [#]');ylabel('Accuracy [%]')
legend(('CL net', 'CNN'))
title('2-layer networks')
show(block=False)
savefig('fig-svhn-clnet.eps')