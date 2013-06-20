require 'image'

test = torch.load('scratch/stopPedBckCarSTOP-images/test.t7')
boundaries = {1,564,1104,1938,2206}
h = 24
v = 4
imageCollection = torch.Tensor(3,46*4*v,46*h)
for j = 1, 46*h, 46 do
   for k = 0, 3 do
      for i = 1, 46*v, 46 do
         imageCollection[{ {},{i+v*k*46,i+v*k*46+46-1},{j,j+46-1} }] = test.data[math.random(boundaries[k+1],boundaries[k+2]-1)]
      end
   end
end

image.display(imageCollection)

io.write('Save (y/n)? ')
if io.read() == 'y' then
   io.write('Name (.png): ')
   image.savePNG(io.read(),imageCollection)
end
