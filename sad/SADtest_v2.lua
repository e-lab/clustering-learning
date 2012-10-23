-- May 10th 2012
-- EC test on sum of absolute differences (SAD) algorithm
-- test on TLD video dataset: search-region bounded algorithm (more efficient)

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
--filename = '/Users/eugenioculurciello/AdvancedResearch/SyntheticVision/datasets/TLD/07_motocross/'
--inputim=image.loadPNG(filename..'00001.png',3)
---- initial object/patch location and video frame
--px=300
--py=40
--src_rng=100
--i=1
filename = '/Users/eugenioculurciello/AdvancedResearch/SyntheticVision/datasets/TLD/05_pedestrian3/'
inputim=image.loadJPG(filename..'00001.jpg',3)
-- initial object/patch location and video frame
px=160
py=120
src_rng=50
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

-- search range for SAD maxima/min
if src_rng < fil2r then print("Error: search range src_rng < patch size/2") end

-- inits:
processed=torch.zeros(ir,ic)
outim=torch.zeros(inputim:size())
ker=torch.zeros(inputim:size(1), fil_r, fil_c)
ker=image.crop(Ninputim, px-fil2c, py-fil2r, px+fil2c+1, py+fil2r+1)

-- loop on video frames:
function process()
	--inputim=image.loadPNG(string.format(filename..'%0005d', i) .. '.png',3)
	inputim=image.loadJPG(string.format(filename..'%0005d', i) .. '.jpg',3)
	print('Frame count:'..i)

	-- global linear normalization of input frame
	kNorm = torch.ones(5) --gaussian = image.gaussian(3,0.15)
	m=nn.SpatialSubtractiveNormalization(3,kNorm)
	Ninputim = m:forward(inputim)
	
	--ker=image.crop(Ninputim, px-fil2c, py-fil2r, px+fil2c+1, py+fil2r+1) -- test ONLY!!!!
	-- SAD: REMEMBER TO INIT OUTput to 0!!!
	processed=processed:mul(0)
	
	y1=math.max(py-src_rng,1) -- inputs box: bounded (src_rng) around (px,py) 
	y2=math.min(py+src_rng,ir)
	x1=math.max(px-src_rng,1)
	x2=math.min(px+src_rng,ic)
	
	tim=image.crop(Ninputim, x1,y1,x2,y2) -- input patch if input image
		
	nir = tim:size(2)
	nic = tim:size(3)
	nour = (nir - kr) + 1
	nouc = (nic - kc) + 1
	
	ym = y1+fil2r -- outputs box (shrinked from input box)
	yM = y2-fil2r-1
	xm = x1+fil2c
	xM = x2-fil2c-1
	
	total=torch.zeros(nour, nouc) -- outputs
	temp1=torch.zeros(nour, nouc)
	temp2=torch.zeros(nour, nouc)
	temp3=torch.zeros(nour, nouc)

	SADbw(tim[1],ker[1],temp1)
	SADbw(tim[2],ker[2],temp2)
	SADbw(tim[3],ker[3],temp3)
	
	total=temp1+temp2+temp3
	total=total/torch.max(total)
	total = total:mul(-1):add(1)

	processed[{{ym,yM},{xm,xM}}] = total
	outim[1]:copy(processed)
	outim[2]:copy(processed)
	outim[3]:copy(processed)
	
	-- update object/patch location
	value, px_nxt, py_nxt = GetMax(total)
	--print('SAD output:', value)
	px = px_nxt + xm -1 
    py = py_nxt + ym -1 
       
    -- next patch extraction
	ker=image.crop(Ninputim, px-fil2c, py-fil2r, px+fil2c+1, py+fil2r+1)
    
    --if i==1 then image.display(total) end
    -- continue loop, chose endframe
    if i==100 then i=1 else i=i+1 end
    -- close program window at a certain frame if needed:
    --if i==10 then exit() end
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

	


