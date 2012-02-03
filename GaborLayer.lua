-- A simple code for gabor filters
-- author: Prashant Lalwani Fall 2011

--Description:

-- Real part of a gabor filter: 
--	g(x,y,lambda,theta,shi,sigma,gamma) = exp{-(xPrime^2 + gamma^2 * yPrima^2)/(2 * sigma^2)} * cosine(2*pi*xPrime/lambda + shi)
-- 	xPrime =  xcos(theta) + ysin(theta)
-- 	yPrime = -xsin(theta) + ycos(theta)

-- 	lambda represents the wavelength of the filter, values above 2 and less than 1/5 of image size are valid
		--	Its value is specified in pixels

--	theta represents orientation of the normal to the paralle stripes of a gabor function
		--	Its value is specified in degrees so the valid range becomes 0-360

--	shi is the phase offset, valid values are real numbers between -180 and 180
		--	The values 0 and 180 correspond to center-symmetric center-on and center-off functions respectively
		-- 	It can be used to change the intensity of the background with respect to the filter

--	sigma is the gaussian envelope, sigma = 0.56*lambda should work in our case which gives a bandwidth of 1
		--	sigma = 0.56*lambda = 4.48 for a bandwidth of 1 so 1/(2*sigma^2) = 0.02 as used in equation 3

--	gamma is the spatial aspect ratio and defines the ellipticity of the filter i.e. gamma = 1 will give a circle


-- Gabor filter algorithm to compute the Tensor
function GaborLayer(Sx,Sy,lambda,theta)
	sigma = 0.56*lambda
	Gabor = torch.Tensor(Sx,Sy)
	for x = 1,Sx do
		for y = 1,Sy do
	xPrime =  (x-Sx/2-1)*math.cos(theta) + (y-Sy/2-1)*math.sin(theta)	--equation 1
	yPrime = -(x-Sx/2-1)*math.sin(theta)  + (y-Sy/2-1)*math.cos(theta)	--equation 2
	 Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.cos(2*pi*xPrime/lambda  + shi)	-- equation 3 	
	end
		end	
	return(Gabor)
end
