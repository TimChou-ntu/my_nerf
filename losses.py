import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
    
class CrossEntrypyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs["semantic_final"], targets)
        
        return loss

# NeSF use also regularization term on neighboring area to have similar semantic class
loss_dict = {'mse': MSELoss, 'crossentropy': CrossEntrypyLoss}