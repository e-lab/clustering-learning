-- May 10th 2012
-- EC test on sum of absolute differences (SAD) algorithm
-- this should eb run on patches with high info content
-- DO NOT run on uniform patches, or you GET NOTHING GOOD!!!

require 'nn'
require 'image'
require 'inline'

-- do everything in single precision
torch.setdefaulttensortype('torch.FloatTensor')

-- some coroutines
local c = {}

-- some defines
inline.preamble [[
      #define max(a,b)  ((a)>(b) ? (a) : (b))
      #define abs(a)    (a) < 0 ? -(a) : (a)
      #define square(a) (a)*(a)
]]

SAD = inline.load [[
      // get args
      const void* id = luaT_checktypename2id(L, "torch.FloatTensor");
      THFloatTensor *input = luaT_checkudata(L, 1, id);
      THFloatTensor *kernel = luaT_checkudata(L, 2, id);
      THFloatTensor *output = luaT_checkudata(L, 3, id);
      
      // get raw pointers
      float *input_data = THFloatTensor_data(input);
      float *kernel_data = THFloatTensor_data(kernel);
      float *output_data = THFloatTensor_data(output);
      
      // dims
      int ir = input->size[1];
      int ic = input->size[2];
      int kr = kernel->size[1];
      int kc = kernel->size[2];
      int or = (ir - kr) + 1;
      int oc = (ic - kc) + 1;

      // sum of absolute differences (SAD)
      int xx,yy,kx,ky;
      for(yy = 0; yy < or; yy++) {
        for(xx = 0; xx < oc; xx++) {
          /* SAD (input image vs kernel) */
          float *pi_ = input_data + yy*ic + xx;
          float *pw_ = kernel_data;
          float sum = 0;
          for(ky = 0; ky < kr; ky++) {
            for(kx = 0; kx < kc; kx++) {
              sum += abs(pi_[kx]-pw_[kx]);
            }
            pi_ += ic; /* next input line */
            pw_ += kc; /* next mask line */
          }
          /* Update output */
          *output_data++ += sum;
        }
      }  
      return 0;
]]

-- input image/test:
lena=image.lena()
-- global linear normalization of input frame
kNorm = torch.ones(9)
m=nn.SpatialSubtractiveNormalization(3,kNorm)
Nlena = m:forward(lena)

-- kernel/patch size
fil_s = 31
ker=torch.zeros(lena:size(1),fil_s,fil_s)

-- get sizes:
ir = Nlena:size(2)
ic = Nlena:size(3)
kr = fil_s
kc = fil_s
our = (ir - kr) + 1
ouc = (ic - kc) + 1

processed = torch.zeros(our,ouc)

-- patch location
px=100
py=100
-- patch extraction
ker=image.crop(Nlena, px, py, px+fil_s, py+fil_s)
-- SAD:
SAD(Nlena,ker,processed)
print(torch.min(processed), torch.max(processed))
processed=processed/torch.max(processed)
print(torch.min(processed), torch.max(processed))
processed= processed:mul(-1):add(1)
print(torch.min(processed), torch.max(processed))
-- show a pseudo-bounding box: 4 dots on screen
processed[py+(fil_s-1)/2][px+(fil_s-1)/2]=1
processed[py+(fil_s-1)/2][px-(fil_s-1)/2]=1
processed[py-(fil_s-1)/2][px+(fil_s-1)/2]=1
processed[py-(fil_s-1)/2][px-(fil_s-1)/2]=1

-- display: returns 1 (white) in highly similar patches
w1=image.display(processed)
	
	


