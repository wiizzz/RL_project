
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gymnasium as gym
import torch as th

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LayerNorm([3, 100, 156]),
                                        
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.LayerNorm([32, 24, 38]),
            nn.LeakyReLU(**kwargs),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm([64, 11, 18]),
            nn.LeakyReLU(**kwargs),
                                                                                                                
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 9, 16]),
            nn.LeakyReLU(**kwargs),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(9216, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(**kwargs),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        self.cnn = nn.Sequential(
            nn.LayerNorm([3, 100, 156]),
                                        
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.LayerNorm([32, 24, 38]),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm([64, 11, 18]),
            nn.LeakyReLU(),
                                                                                                                
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 9, 16]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        '''
        self.linear = nn.Sequential(
            nn.Linear(9216, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        '''
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channe===l (subspace.shape[0] == 0)
                
                extractors[key] = self.cnn
                total_concat_size += 128 #subspace.shape[1] // 4 * subspace.shape[2] // 4
                # print("size", subspace.shape[1] // 4 * subspace.shape[2] // 4)
                # print("=================================fuckyou", extractors[key])
            elif key == "vector":
                # Run through a simple MLP
                # print("=================================\n",subspace.shape[0])
                extractors[key] = nn.Linear(subspace.shape[0], 2)
                # print("fuck")
                total_concat_size += 2

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # print("|||||||||||||||||||||||||||||||||||")
        return th.cat(encoded_tensor_list, dim=1)
