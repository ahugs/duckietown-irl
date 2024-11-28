from abc import ABC

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.common import (
    MLP,
)

def getattr_with_matching_alt_value(obj, attr_name, alt_value):
    """Gets the given attribute from the given object or takes the alternative value if it is not present.
    If both are present, they are required to match.

    :param obj: the object from which to obtain the attribute value
    :param attr_name: the attribute name
    :param alt_value: the alternative value for the case where the attribute is not present, which cannot be None
        if the attribute is not present
    :return: the value
    """
    v = getattr(obj, attr_name)
    if v is not None:
        if alt_value is not None and v != alt_value:
            raise ValueError(
                f"Attribute '{attr_name}' of {obj} is defined ({v}) but does not match alt. value ({alt_value})",
            )
        return v
    else:
        if alt_value is None:
            raise ValueError(
                f"Attribute '{attr_name}' of {obj} is not defined and no fallback given",
            )
        return alt_value


def get_output_dim(module: nn.Module, alt_value) -> int:
    """Retrieves value the `output_dim` attribute of the given module or uses the given alternative value if the attribute is not present.
    If both are present, they must match.

    :param module: the module
    :param alt_value: the alternative value
    :return: the value
    """
    return getattr_with_matching_alt_value(module, "output_dim", alt_value)

class Reward(nn.Module, ABC):
    """Simple reward network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.
    :param linear_layer: use this module as linear layer.
    :param output_activation: use this output activation function
    :param flatten_input: whether to flatten input data for the last layer.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net,
        hidden_sizes= (),
        device = "cpu",
        preprocess_net_output_dim = None,
        output_activation= nn.Identity,
        output_scaling = 1.0,
        clip_range = [-np.inf, np.inf],
        output_transform = None,
        initialize_zero = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            1,
            hidden_sizes,
            device=self.device,        )
        self.output_activation = output_activation()
        assert not (output_scaling != 1.0 and output_transform is not None)
        if output_scaling != 1.0:
            self.output_transform = lambda x: x * output_scaling
        elif output_transform is not None:
            self.output_transform = output_transform
        else: 
            self.output_transform = nn.Identity()
        if clip_range is None:
            self.clip_range = [-np.inf, np.inf]
        else:
            self.clip_range = clip_range

        if initialize_zero:
            self.last.model[0].weight.data.fill_(0)
            self.last.model[0].bias.data.fill_(0)

    def forward(
        self,
        obs,
        info=None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        values_B, hidden_BH = self.preprocess(obs)
        output = self.output_transform(self.output_activation(self.last(values_B)))
        return torch.clamp(output, min=self.clip_range[0], max=self.clip_range[1])
    