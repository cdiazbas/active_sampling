import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import init

"""
A general-purpose residual network for 1-dim inputs.
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LayerNorm(nn.Module):
    # https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LinearNorm(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self,input_size, output_size,activation=F.relu):
        super().__init__()
        self.activation = activation
        self.ln = LayerNorm(output_size)
        self.linear_layer = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        output = self.linear_layer(inputs)
        output = self.activation(output)
        output = self.ln(output)
        return output




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        use_layer_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_norm_layers = nn.ModuleList(
                [LayerNorm(features) for _ in range(2)]
            )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        if self.use_layer_norm:
            temps = self.ln_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        if self.use_layer_norm:
            temps = self.ln_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""
    def __init__(self, in_features, out_features, hidden_features, context_features=None, num_blocks=2, activation=F.elu, dropout_probability=0.0, use_batch_norm=False, use_layer_norm=False, database_input=None, database_output=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
        if database_input is not None and database_output is not None:
            self.mean_input = torch.mean(database_input,axis=0)
            self.std_input = torch.std(database_input,axis=0)
            self.mean_output = torch.mean(database_output,axis=0)
            self.std_output = torch.std(database_output,axis=0)
        else:
            self.mean_input = 0.0
            self.std_input = 1.0
            self.mean_output = 0.0
            self.std_output = 1.0


    def forward_mu(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        return self.output_mu(temps)

    def forward(self, x):
        x = (x-self.mean_input)/self.std_input
        return self.forward_mu(x)*self.std_output + self.mean_output
