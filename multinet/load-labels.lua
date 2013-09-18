--------------------------------------------------------------------------------
-- Constructing datasets' labels
--------------------------------------------------------------------------------

local signLabels
if opt.German then
   signLabels = {
      '20 km/h speed limit',
      '30 km/h speed limit',
      '50 km/h speed limit',
      '60 km/h speed limit',
      '70 km/h speed limit',
      '80 km/h speed limit',
      '80 km/h end of speed limit',
      '100 km/h speed limit',
      '120 km/h speed limih',
      'No passing',
      'No passing for vehicles over 3.5t',
      'Priority',
      'Priority road',
      'Yield',
      'Stop',
      'Prohibited for all vehicles',
      'Vehicles over 3.5t prohibited',
      'Do not enter',
      'General danger',
      'Curve (left)',
      'Curve (right)',
      'Double curve. First curve is to the left',
      'Rough road',
      'Slippery when wet or dirty',
      'Road narrows (right side)',
      'Road work',
      'Traffic signals ahead',
      'Pedestrians',
      'Watch for children',
      'Bicycle crossing',
      'Beware of ice/snow',
      'Wild animal crossing',
      'End of all restrictions',
      'All traffic must turn right',
      'All traffic must turn left',
      'All traffic must continue straight ahead (i.e. no turns)',
      'All traffic must continue straight ahead or turn right (i.e. no left turn)',
      'All traffic must continue straight ahead or turn left (i.e. no right turn)',
      'Pass by on right',
      'Pass by on left',
      'Roundabout',
      'End of no passing zone',
      'End of no passing zone for vehicles over 3.5t'
   }
elseif opt.dataStop then
   signLabels = {'Stop'}
else
   signLabels = {
      'Stop',
      'Signal ahead',
      'Pedestrian crossing',
      'Speed limit 35',
      'Keep right',
      'Speed limit urdbl',
      'Merge',
      'School',
      'Speed limit 25',
      'Added lane',
      'Stop ahead',
      'Speed limit 40',
      'Speed limit 45',
      'Yield ahead',
      'School speed limit 25',
      'Speed limit 50',
      'Yield',
      'Right lane must turn',
      'Turn right',
      'Truck speed limit 55',
      'Lane ends',
      'Speed limit 65',
      'Ramp speed advisory 50',
      'Ramp speed advisory 45',
      'Do not pass',
      'No left turn',
      'Slow',
      'Zone ahead 45',
      'No right turn',
      'Turn left',
      'Ramp speed advisory urdbl',
      'Dip',
      'Speed limit 55',
      'Intersection',
      'Ramp speed advisory 20'
   }
end

local siftflowLabels = torch.load('siftflowLabels.t7')
if opt.siftflow then
   classes = siftflowLabels
else
   local k
   classes = {}
   if opt.includeSiftflow then
      for i = 1, nbClasses[1] do classes[#classes+1] = siftflowLabels[i] end
      k = 3
   else k = 2
   end
   table.insert(classes,'Person')
   for i = 1, nbClasses[k]  do classes[#classes+1] = signLabels[i] end
   classes[#classes+1] = 'Background'
   classes[#classes+1] = 'Car'
end

