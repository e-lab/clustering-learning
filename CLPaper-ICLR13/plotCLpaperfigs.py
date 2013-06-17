
# Plotter of test/train data for CL paper
# E. Culurciello
# October 2012

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
cy1 = load_data_file('results/ayse-optimized/cifar-CL-1st.log')
cx1 = range(0,50)#len(cy1))
cy2 = load_data_file('results/ayse-optimized/cifar-CL-2nd.log')
#cx2 = range(0,len(cy2))
cy3 = load_data_file('results/cifar-convnet-1L-16/test.log')
cx3 = range(0,len(cy3))
cy4 = load_data_file('results/cifar-convnet-16-128/test.log')
#cx4 = range(0,len(cy4))

figure(0)
plot(cx1, cy1[0:50], 'g--', cx1, cy2[0:50], 'g-', cx3, cy3, 'b--', cx1, cy4[0:50] ,'b-')
xlabel('epoch [#]');ylabel('Accuracy [%]')
legend(('CL 1 layer', 'CL 2 layers', 'CNN 1l', 'CNN 2l'))
#title('data_cifar.eps')
#show(block=False)



# load all needed data: SVHN
sy1 = load_data_file('results/ayse-optimized/svn-CL-1st.log')
sx1 = range(0,len(sy1))
sy2 = load_data_file('results/ayse-optimized/svn-CL-2nd.log')
sx2 = range(0,len(sy2))
sy3 = load_data_file('results/svhn-convnet-1L-16/test.log')
sx3 = range(0,len(sy3))
sy4 = load_data_file('results/svhn-convnet-16-128/test.log')
sx4 = range(0,len(sy4))

figure(1)
plot(sx1, sy1, 'g--', sx2, sy2, 'g-', sx3, sy3, 'b--', sx4, sy4,'b-')
xlabel('epoch [#]');ylabel('Accuracy [%]')
legend(('CL 1 layer', 'CL 2 layers', 'CNN 1l', 'CNN 2l'))
#title('data_svhn.eps')
show(block=False)