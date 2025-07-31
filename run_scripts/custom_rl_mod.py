import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict, Any, Mapping
import numpy as np

class CustomCNNTorchRLModule(TorchRLModule):
    """Custom TorchRLModule to handle flattened dictionary observations with CNN for images and FC for agent data."""
    
    def __init__(self, observation_space=None, action_space=None, inference_only=False, 
                 learner_only=False, model_config=None, **kwargs):
        # Use the new RLLib API format
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            **kwargs
        )
        
        obs_space = observation_space
        model_config = model_config or {}
        
        # Setup observation structure from model config
        self._setup_obs_structure(model_config)
        
        # Verify observation space matches expected size
        expected_total_size = self.image_size + self.agent_features_size
        if obs_space.shape[0] != expected_total_size:
            raise ValueError(f"Observation space size {obs_space.shape[0]} doesn't match "
                           f"expected size {expected_total_size}")
        
        # Build CNN for processing map image
        self._build_cnn_layers(model_config)
        
        # Build FC layers for agent features (if they exist)
        self._build_agent_fc_layers(model_config)
        
        # Build post-processing layers
        self._build_post_layers(model_config)
        
        # Q-value head
        self.q_value_head = nn.Linear(self.feature_size, action_space.n)
        
    def _setup_obs_structure(self, model_config):
        """Setup observation structure from model config."""
        custom_config = model_config.get("custom_model_config", {})
        
        # Image observation parameters
        self.view_len = custom_config.get("view_len", 7)
        self.image_shape = (2 * self.view_len + 1, 2 * self.view_len + 1, 3)
        self.image_size = int(np.prod(self.image_shape))
        
        # Agent features parameters
        self.has_agent_actions = custom_config.get("return_agent_actions", False)
        if self.has_agent_actions:
            self.num_agents = custom_config.get("num_agents", 2)
            self.agent_actions_size = self.num_agents - 1
            self.visible_agents_size = self.num_agents - 1
            self.prev_visible_size = self.num_agents - 1
            self.agent_features_size = (self.agent_actions_size + 
                                     self.visible_agents_size + 
                                     self.prev_visible_size)
        else:
            self.agent_features_size = 0
        
        # Create indices for extracting components from flattened observation
        self._create_obs_indices()
    
    def _create_obs_indices(self):
        """Create indices for extracting components from flattened observation."""
        self.obs_indices = {}
        current_idx = 0
        
        # Image indices
        self.obs_indices["curr_obs"] = (current_idx, current_idx + self.image_size)
        current_idx += self.image_size
        
        # Agent feature indices
        if self.has_agent_actions:
            self.obs_indices["other_agent_actions"] = (current_idx, current_idx + self.agent_actions_size)
            current_idx += self.agent_actions_size
            
            self.obs_indices["visible_agents"] = (current_idx, current_idx + self.visible_agents_size)
            current_idx += self.visible_agents_size
            
            self.obs_indices["prev_visible_agents"] = (current_idx, current_idx + self.prev_visible_size)
    
    def _build_cnn_layers(self, model_config):
        """Build CNN layers for image processing."""
        conv_filters = model_config.get("conv_filters", [[32, [3, 3], 1], [64, [3, 3], 1]])
        conv_activation = model_config.get("conv_activation", "relu")
        
        self.conv_layers = nn.Sequential()
        in_channels = self.image_shape[2]  # RGB channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(conv_filters):
            padding = 1 if isinstance(kernel_size, int) else kernel_size[0] // 2
            self.conv_layers.add_module(
                f"conv_{i}",
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
            )
            if conv_activation == "relu":
                self.conv_layers.add_module(f"relu_{i}", nn.ReLU())
            elif conv_activation == "tanh":
                self.conv_layers.add_module(f"tanh_{i}", nn.Tanh())
            in_channels = out_channels
        
        # Calculate CNN output size
        self._calculate_cnn_output_size()
    
    def _calculate_cnn_output_size(self):
        """Calculate the output size of CNN layers."""
        with torch.no_grad():
            # Create dummy input with proper shape
            dummy_input = torch.zeros(1, self.image_shape[2], 
                                    self.image_shape[0], self.image_shape[1])
            cnn_output = self.conv_layers(dummy_input.float())
            self.cnn_output_size = int(np.prod(cnn_output.shape[1:]))
    
    def _build_agent_fc_layers(self, model_config):
        """Build FC layers for agent features."""
        if self.has_agent_actions:
            agent_fc_hiddens = model_config.get("agent_fc_hiddens", [64])
            
            self.agent_fc_layers = nn.Sequential()
            prev_size = self.agent_features_size
            for i, hidden_size in enumerate(agent_fc_hiddens):
                self.agent_fc_layers.add_module(
                    f"agent_fc_{i}", 
                    nn.Linear(prev_size, hidden_size)
                )
                self.agent_fc_layers.add_module(f"agent_relu_{i}", nn.ReLU())
                prev_size = hidden_size
            
            self.agent_fc_output_size = prev_size
        else:
            self.agent_fc_output_size = 0
    
    def _build_post_layers(self, model_config):
        """Build post-processing layers."""
        combined_size = self.cnn_output_size + self.agent_fc_output_size
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [256])
        post_fcnet_activation = model_config.get("post_fcnet_activation", "relu")
        
        self.post_layers = nn.Sequential()
        prev_size = combined_size
        for i, hidden_size in enumerate(post_fcnet_hiddens):
            self.post_layers.add_module(
                f"post_fc_{i}",
                nn.Linear(prev_size, hidden_size)
            )
            if post_fcnet_activation == "relu":
                self.post_layers.add_module(f"post_relu_{i}", nn.ReLU())
            elif post_fcnet_activation == "tanh":
                self.post_layers.add_module(f"post_tanh_{i}", nn.Tanh())
            prev_size = hidden_size
        
        self.feature_size = prev_size
    
    def _reconstruct_obs_from_flattened(self, flattened_obs: TensorType) -> Dict[str, TensorType]:
        """Reconstruct dictionary-like structure from flattened observation."""
        batch_size = flattened_obs.shape[0]
        reconstructed_obs = {}
        
        # Extract image
        start_idx, end_idx = self.obs_indices["curr_obs"]
        image_flat = flattened_obs[:, start_idx:end_idx]
        reconstructed_obs["curr_obs"] = image_flat.reshape(
            batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]
        )
        
        # Extract agent features if they exist
        if self.has_agent_actions:
            for key in ["other_agent_actions", "visible_agents", "prev_visible_agents"]:
                if key in self.obs_indices:
                    start_idx, end_idx = self.obs_indices[key]
                    reconstructed_obs[key] = flattened_obs[:, start_idx:end_idx]
        
        return reconstructed_obs
    
    def _process_image_obs(self, image_obs: TensorType) -> TensorType:
        """Process image observation through CNN."""
        # Input shape: (batch_size, height, width, channels)
        # Convert to (batch_size, channels, height, width) for CNN
        if len(image_obs.shape) == 3:  # Single observation without batch dimension
            image_obs = image_obs.unsqueeze(0)  # Add batch dimension
        
        # Convert from (B, H, W, C) to (B, C, H, W)
        if image_obs.shape[-1] == 3:  # Last dimension is channels
            image_obs = image_obs.permute(0, 3, 1, 2)
        
        return self.conv_layers(image_obs.float())
    
    def _process_agent_features(self, agent_features: TensorType) -> TensorType:
        """Process agent-related features through FC layers."""
        if not self.has_agent_actions or agent_features is None:
            return None
        
        return self.agent_fc_layers(agent_features.float())
    
    def _forward_train(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """Forward pass for training (same as inference for DQN)."""
        return self._forward_inference(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """Forward pass for inference."""
        obs = batch[SampleBatch.OBS]
        
        # Handle different observation formats
        if isinstance(obs, (list, tuple)):
            # Sequence of observations - stack them
            stacked_obs = torch.stack([
                torch.tensor(o, dtype=torch.float32) if not isinstance(o, torch.Tensor) else o.float() 
                for o in obs
            ])
            flattened_obs = stacked_obs
        elif isinstance(obs, torch.Tensor):
            flattened_obs = obs.float()
        else:
            flattened_obs = torch.tensor(obs, dtype=torch.float32)
        
        # Ensure batch dimension
        if len(flattened_obs.shape) == 1:
            flattened_obs = flattened_obs.unsqueeze(0)
        
        # Reconstruct dictionary structure from flattened observation
        reconstructed_obs = self._reconstruct_obs_from_flattened(flattened_obs)
        
        # Process image through CNN
        cnn_output = self._process_image_obs(reconstructed_obs["curr_obs"])
        cnn_features = cnn_output.reshape(cnn_output.size(0), -1)  # Flatten
        
        # Process agent features if they exist
        if self.has_agent_actions:
            # Concatenate all agent features
            agent_features_list = []
            for key in ["other_agent_actions", "visible_agents", "prev_visible_agents"]:
                if key in reconstructed_obs:
                    agent_features_list.append(reconstructed_obs[key])
            
            if agent_features_list:
                agent_features_combined = torch.cat(agent_features_list, dim=-1)
                agent_features_processed = self._process_agent_features(agent_features_combined)
                
                # Combine CNN and agent features
                combined_features = torch.cat([cnn_features, agent_features_processed], dim=1)
            else:
                combined_features = cnn_features
        else:
            combined_features = cnn_features
        
        # Post-processing
        features = self.post_layers(combined_features)
        
        # Q-values
        q_values = self.q_value_head(features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: q_values,
        }
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """Forward pass for exploration (same as inference for DQN)."""
        return self._forward_inference(batch, **kwargs)
    
    def get_q_values(self, batch: Dict[str, TensorType], **kwargs) -> TensorType:
        """Get Q-values for the given batch."""
        return self._forward_inference(batch, **kwargs)[SampleBatch.ACTION_DIST_INPUTS]
    
    def get_initial_state(self) -> Dict[str, TensorType]:
        """Get initial state (empty for DQN as it's not recurrent)."""
        return {}