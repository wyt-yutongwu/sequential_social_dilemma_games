import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class DQN_Module(nn.Module):

    def __init__(self, env, conv_filters, fc_filters, action_shape,obs_shape,use_cnn = False):
        super(DQN_Module, self).__init__()
        obs, info = env.reset()
        self.conv_filters = conv_filters
        self.fc_filters = fc_filters
        agent_ids = list(env.agents.keys())
        # self.obs_shape = len(obs)
        # self.action_shape = env.action_space.n
        self.obs_shape = obs_shape#obs[agent_ids[0]].shape
        print(f"obs: {self.obs_shape}")
        print(f"action_space: {env.action_space.n}")
        self.action_shape = action_shape#env.action_space.n
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.build_cnn_layers()
        else:
            self.cnn_out_size = self.obs_shape
            # self.cnn_out_size = self.obs_shape[0]
        self.build_fc_layers()
        self.output_layer = nn.Linear(self.fc_out_size, self.action_shape)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if self.use_cnn:
            x = x.permute(0, 3, 1, 2)
            x = torch.flatten(self.conv_layers(x),start_dim=1)
        x = self.fc_layers(x)

        x= self.output_layer(x)#.unsqueeze(0)

        return x
    
    def _get_conv_out(self, shape):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            # Create dummy input with batch size 1
            dummy_input = torch.zeros(1, *shape)  # (1, 3, 5, 5)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.size()))
        
    def build_cnn_layers(self):
        H,W,C = self.obs_shape
        out_size = H
        self.prev_in = C
        self.conv_layers = nn.Sequential()
        count = 0
        for layers in self.conv_filters:
            out_chan, kernel, stride = layers
            self.conv_layers.add_module(f"conv_{count}",
                                        nn.Conv2d(  in_channels=self.prev_in,
                                        out_channels=out_chan, 
                                        kernel_size=(kernel[0], kernel[1]),
                                        stride=stride,
                                        padding=0))
            self.conv_layers.add_module(f"relu_{count}", nn.ReLU())
            out_size = (out_size - kernel[0]) // stride + 1

            self.prev_in = out_chan
            count += 1
        print("added conv layers", self.conv_layers)
        self.cnn_out_size = self._get_conv_out((C,H,W))
    
    def build_fc_layers(self):
        self.fc_layers = nn.Sequential()
        prev_in = self.cnn_out_size
        count = 0
        for layers in self.fc_filters:
            self.fc_layers.add_module(f"fc_{count}",
                                      nn.Linear(prev_in, layers))
            self.fc_layers.add_module(f"relu_{count}", nn.ReLU())
            count += 1
            prev_in = layers
        self.fc_out_size = prev_in
