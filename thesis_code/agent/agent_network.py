import torch
from torch import nn
import numpy as np

class AgentNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, freeze=False):
        super().__init__()
        # Conv block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self._calculate_conv_output_size(input_shape)

        # Fully connected block
        self.fc_layers = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        if freeze:
            self._freeze_layers()
        
        # Using CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.fc_layers(x)

    def _calculate_conv_output_size(self, shape):
        conv_output = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(conv_output.size()))
    
    def _freeze_layers(self):        
        for param in self.fc_layers.parameters():
            param.requires_grad = False
