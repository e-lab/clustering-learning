-- May 10th 2012
-- EC test on sum of absolute differences (SAD) algorithm
-- test on TLD video dataset: full frame processing

-- motocross lost on frame 27 with a 31x31 kernel at y_init,x_init=40,300

require 'nn'
require 'image'
require 'inline'

require 'qt'
require 'qtwidget'
require 'qtuiloader'

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


SADbw = inline.load [[
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
      int ir = input->size[0];
      int ic = input->size[1];
      int kr = kernel->size[0];
      int kc = kernel->size[1];
      int or = (ir - kr) + 1;
      int oc = (ic - kc) + 1;

      // sum of absolute differences (SAD)
      int xx,yy,kx,ky;
      for(yy = 0; yy < or; yy++) {
        for(xx = 0; xx < oc; xx++) {
          /* SAD (input image vs kernel) */
          float *pi_ = input_data + yy*ic + xx;
          float *pw_ = kernel_data;
          float *pw1_ = kernel_data + kr*kc;
          float *pw2_ = kernel_data + 2*kr*kc;
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

SADcolor = inline.load [[
      // get args
      const void* id = luaT_checktypename2id(L, "torch.FloatTensor");
      THFloatTensor *input = luaT_checkudata(L, 1, id);
      THFloatTensor *kernel = luaT_checkudata(L, 2, id);
      THFloatTensor *output = luaT_checkudata(L, 3, id);
      
      // get raw pointers
      float *input_data = THFloatTensor_data(input);
      float *kernel_data = THFloatTensor_data(kernel);
      float *output_data = THFloatTensor_data(output);
      
	  //get dims:
	  int it = input->size[0];
	  int ir = input->size[1];
      int ic = input->size[2];
      int kt = kernel->size[0];
      int kr = kernel->size[1];
      int kc = kernel->size[2];
	  int ot = output->size[0]-1;
	  //printf("o=%g ", ot);
	  int or = output->size[1];//(ir - kr) + 1;
	  int oc = output->size[2];//(ic - kc) + 1;
	
	  int zz, xx, yy;
	  
	  for (zz = 0; zz < ot; zz++)
	  {
	    for(yy = 0; yy < or; yy++)
	    {
	      for(xx = 0; xx < oc; xx++)
	      {
	        /* Dot product in two dimensions... (between input image and the mask) */
	        float *pi_ = input_data + zz*ir*ic + yy*ic + xx;
	        float *pw_ = kernel_data;
	        float sum = 0;
	        int kz, kx, ky;
	        for(kz = 0; kz < kt; kz++)
	        {
	          for(ky = 0; ky < kr; ky++)
	          {
	            for(kx = 0; kx < kc; kx++) {
	              sum += abs(pi_[kx]-pw_[kx]);
	            }
	            pi_ += ic; /* next input line */
	            pw_ += kc; /* next mask line */
	          }
	        }
	        /* Update output */
	        *output_data++ += sum;
	      }
	    }
	  }
      return 0;
]]

-- function max
function GetMax(a)
	x,xi = torch.max(a,1)
	y,yi = torch.max(x,2) -- y = value

	x_out = yi[1][1]  -- y coord
	y_out = xi[1][x_out] -- x coord
	return y,x_out,y_out 
end

-- setup GUI (external UI file)
if not win or not widget then 
   widget = qtuiloader.load('g.ui')
   win = qt.QtLuaPainter(widget.frame) 
end

-- profiler
p = xlua.Profiler()

-- input image/test:
filename = '/Users/eugenioculurciello/AdvancedResearch/SyntheticVision/datasets/TLD/07_motocross/'
inputim=image.loadPNG(filename..'00001.png',3)
-- initial object/patch location and video frame
px=300
py=40
i=1

-- global linear normalization of input frame
kNorm = torch.ones(5)
m=nn.SpatialSubtractiveNormalization(3,kNorm)
Ninputim = m:forward(inputim)

-- kernel/patch size
fil_r = 31 -- has to be odd sized
fil_c = 31
fil2r = (fil_r-1)/2 -- because of this...
fil2c = (fil_c-1)/2

-- get sizes:
ir = Ninputim:size(2)
ic = Ninputim:size(3)
kr = fil_r
kc = fil_c
our = (ir - kr) + 1
ouc = (ic - kc) + 1

-- inits:
processed=torch.zeros(ir,ic)
total=torch.zeros(our, ouc) -- outputs
temp1=torch.zeros(our, ouc)
temp2=torch.zeros(our, ouc)
temp3=torch.zeros(our, ouc)
outim=torch.zeros(inputim:size())
ker=torch.zeros(inputim:size(1), fil_r, fil_c)
ker=image.crop(Ninputim, px-fil2c, py-fil2r, px+fil2c+1, py+fil2r+1)


-- loop on video frames:
function process()
	inputim=image.loadPNG(string.format(
		filename..'%0005d', i) .. '.png',3)

	-- global linear normalization of input frame
	kNorm = torch.ones(5) --gaussian = image.gaussian(3,0.15)
	m=nn.SpatialSubtractiveNormalization(3,kNorm)
	Ninputim = m:forward(inputim)
	
	ym = fil2r -- outputs box (shrinked from input box)
	yM = ir-fil2r-1
	xm = fil2c
	xM = ic-fil2c-1
	--print(ym,yM,xm,xM)

	-- SAD: REMEMBER TO INIT OUTput to 0!!!
	temp1:mul(0)
	temp2:mul(0)
	temp3:mul(0)	
	SADbw(Ninputim[1],ker[1],temp1)
	SADbw(Ninputim[2],ker[2],temp2)
	SADbw(Ninputim[3],ker[3],temp3)

	total=temp1+temp2+temp3
	total=total/torch.max(total)
	total = total:mul(-1):add(1)
	--print(total:size())
	processed[{{ym,yM},{xm,xM}}] = total
	outim[1]:copy(processed)
	outim[2]:copy(processed)
	outim[3]:copy(processed)

	-- update object/patch location
	value, px_nxt, py_nxt = GetMax(processed)
	print('SAD output:', value)
    py=py_nxt
    px=px_nxt

    -- next patch extraction
	ker=image.crop(Ninputim, px-fil2c, py-fil2r, px+fil2c+1, py+fil2r+1)
	
    -- continue loop, chose endframe
    if i==100 then i=1 else i=i+1 end
    print('Frame count:'..i)
end	

-- display function
function display()
   win:gbegin()
   win:showpage()
   -- (1) display input image + pyramid
   --image.display{image=frame, win=win, saturation=false, min=0, max=1}
   image.display{image={inputim,Ninputim,outim}, 
   	  win=win, --saturation=false, min=0, max=1,
      legends={'Input','Normed', 'processed'}}

   -- (2) overlay bounding boxes for each detection
   win:setcolor(1,0,0)
   win:rectangle(px-fil2c, py-fil2r, fil_c, fil_r)
   win:rectangle(ic+px-fil2c, py-fil2r, fil_c, fil_r)
   win:rectangle(2*ic+px-fil2c, py-fil2r, fil_c, fil_r)
   win:stroke()
   --win:setfont(qt.QFont{serif=false,italic=false,size=16})
   --win:moveto(detect.x, detect.y-1)
   --win:show('face')
   win:gend()
   -- wait for key pressed before moving to next frame:
--   while qt.connect(widget.listener,
--      'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
--      function(...) print("MousePress",...) end )==0 do break end
end

-- setup gui
timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              p:start('full loop','fps')
              --p:start('prediction','fps')
              process()
              --p:lap('prediction')
              --p:start('display','fps')
              display()
              --p:lap('display')
              require 'openmp'
              timer:start()
              p:lap('full loop')
              p:printAll()
           end)
widget.windowTitle = 'SAD tracker'
widget:show()
timer:start()