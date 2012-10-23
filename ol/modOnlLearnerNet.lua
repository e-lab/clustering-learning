
-- modification to online learner net to be able to run only with SubSampling, not max pooling

run torch

require 'nnx'
encoder = torch.load('encoder.net')
-- this net has MaxPooling layers in modules 3,6
encoderm = encoder:clone()
-- so I swap them with Spatial Sub Sampling:
spooler3 = nn.SpatialSubSampling(16,2,2,2,2)
spooler6 = nn.SpatialSubSampling(128,2,2,2,2)
-- copy them into new net clone
encoderm.modules[6] = spooler6:clone()
encoderm.modules[3] = spooler3:clone()
encoderm.modules[6].gradInput = encoder.modules[6].gradInput
encoderm.modules[6].output = encoder.modules[6].output
encoderm.modules[3].gradInput = encoder.modules[3].gradInput
encoderm.modules[3].output = encoder.modules[3].output
-- save new net with SpatialSubSampling and not max
torch.save('encoder_new.net',encoderm)