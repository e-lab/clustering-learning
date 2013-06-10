
Tests for the second layer connection, first layer has 32 filters for all tests. 

full connection       accuracy 67% = full connection is obtained by changing the train-cifar-CL2lcolor.lua, instead of using the connection matrix function the code uses the vnet2:add(nn.SpatialConvolution(nk1, nk2,  is, is, cvstepsize,cvstepsize)) which provides with the full connection. 

onetoonemapping 73% = one to one mapping is obtained by setting the fanin to 1 in the code train-cifar-CL2lcolor.lua. 

random connection old %74 = random connect is obtained by modifying the createCoCnx function. Instead of calculating the covariance matrix, it finds random connections by inx = math.random(1,nkp). This code has a bug and it gives improvement, it is left as future work. 

random connection new %72 = fanin is equal to 2. The first column of the connection matrix is filled with (1,32) and the second column is with the random numbers. That way we make sure that we use all the filters and it is more convenient to compare with the connection matrix. 

connection matrix %70 original code.  