
require 'xlua'

op = xlua.OptionParser('%prog [options]')
op:option{'-clean', action='store_true', dest='clean', help='clean all samples'}
op:option{'-hflip', action='store_true', dest='flop', help='flip horizontally'}
op:option{'-rotate', action='store', dest='angle', help='rotate by given angle'}
op:option{'-d', '--dataset', action='store', dest='dataset', help='path to dataset',
          default='../datasets/stanford_dataset'}
opt = op:parse()

if opt.clean then
   print('cleaning up!')
   path = opt.dataset .. '/*/*/*/'
   os.execute('rm -f ' .. path..'flipped-* '.. path .. 'rotated-* ' .. opt.dataset..'/*/cached-*')
elseif opt.angle then
   transfoi = ' -distort SRT ' .. opt.angle .. ' '
   transfom = ' -filter point -interpolate nearest-neighbor -distort SRT ' .. opt.angle .. ' '
   convert = true
   prefix = 'rotated-' .. opt.angle .. '-'
elseif opt.flop then
   transfoi = ' -flop '
   transfom = ' -flop '
   convert = true
   prefix = 'flipped-'
else
   op:help()
end

if convert then
   pathi = opt.dataset .. '/train/Images/someplace'
   pathm = opt.dataset .. '/train/Masks/someplace'
   for file in sys.files(pathi) do
      if file ~= '.' and file ~= '..' and not file:find('flipped-') and not file:find('rotated-') then
         -- Image
         print('applying transform [' .. transfoi .. '] to ' .. file)
         os.execute('convert ' .. pathi .. '/' .. file
                    .. transfoi
                    .. pathi .. '/' .. prefix .. file)

         -- Mask
         file = file:gsub('jpg','png')
         print('applying transform [' .. transfom .. '] to ' .. file)
         os.execute('convert ' .. pathm .. '/' .. file
                    .. transfom
                    .. pathm .. '/' .. prefix .. file)
      end
   end
end
