local SpatialSAD, parent = torch.class('nn.SpatialSAD', 'nn.Module')

function SpatialSAD:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   print('<SpatialSAD> WARNING: module under dev!!!')

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   
   self._templates = torch.Tensor(nOutputPlane,nInputPlane, kW, kH)

   self:reset()
end

function SpatialSAD:reset(stdv)
end

function SpatialSAD:templates(tmplates)
   if tmplates then
      if tmplates:size(1) ~= self.nOutputPlane then
         error('templates() you must provide <nOutputPlane> templates')
      end
      self._templates:copy(tmplates)
      return self
   end
   return self._templates
end

function SpatialSAD:updateOutput(input)
   im_H = input:size(2) - self:templates():size(3) + 1
   im_W = input:size(3) - self:templates():size(4) + 1
   self.output = self.output or torch.Tensor()
   self._input_unfold = self._input_unfold or torch.Tensor()
   self._templates_rep = self._templates_rep or torch.Tensor()
   self._L1s = self._L1s or torch.Tensor()
   self.output:resize(self.nOutputPlane, self.nInputPlane, im_H, im_W)
   for i=1,self.nOutputPlane do
      -- assuming template is 3D (color template) NTx3(colors)xTHxTW
      -- and input is also 3D, but w/ larger width and height 3(colors)xHxW
      self._input_unfold = input:unfold(2,self:templates():size(3),1)
         :unfold(3,self:templates():size(4),1)
      -- input_unfold is now 3x(H-TH+1)x(W-TW+1)xTHxTW, 
      -- with replicated input patches aligned for the template operations
      self._templates_rep = self:templates()[i]:clone()
         :resize(self:templates():size(2), 1, 1, self:templates():size(3), self:templates():size(4))
      self._templates_rep = self._templates_rep:expandAs(self._input_unfold)
      -- template_rep is now also 3x(H-TH+1)x(W-TW+1)xTHxTW, with the same values replicated, 
      -- but still of the same size in memory! it is expanded to have the same size as input_unfold
      -- now this is the math step: compute 1-norm, or SAD, 
      -- or sum of absolute values of the input with the template:
      -- this is done for each location of the input image here in one step!!!!
      self._L1s = self._input_unfold:clone():add(-1,self._templates_rep)
         :abs():sum(5):sum(4):squeeze()--:sum(1):squeeze()
      -- L1s is now a (H-TH+1)x(W-TW+1) tensor, where each entry is the L1 for the given location
      self.output[i] = self._L1s
   end
   return self.output
end

function SpatialSAD:updateGradInput(input, gradOutput)
   error('<SpatialSAD:updateGradInput> not defined yet')
end

function SpatialSAD:accGradParameters(input, gradOutput, scale)
   error('<SpatialSAD:accGradParameters> not defined yet')
end
