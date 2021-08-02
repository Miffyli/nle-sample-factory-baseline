import torch as th
from torch import nn

from sample_factory.algorithms.appo.model_utils import (
    get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, register_custom_encoder
)

from sample_factory.utils.utils import log


class NLEMainEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        # Use standard CNN for the image observation in "obs"
        # See all arguments with "-h" to change this head to e.g. ResNet
        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.vector_obs_head = None
        self.message_head = None
        if 'vector_obs' in obs_shape:
            self.vector_obs_head = nn.Sequential(
                nn.Linear(obs_shape.vector_obs[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            out_size = 128
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size + obs_shape.vector_obs[0]
        if 'message' in obs_shape:
            # _Very_ poor for text understanding,
            # but it is simple and probably enough to overfit to specific sentences.
            self.message_head = nn.Sequential(
                nn.Linear(obs_shape.message[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            out_size = 128
            self.encoder_out_size += out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        # This one handles the "obs" key which contains the main image
        x = self.basic_encoder(obs_dict)

        cats = [x]
        if self.vector_obs_head is not None:
            vector_obs = self.vector_obs_head(obs_dict['vector_obs'].float())
            cats.append(vector_obs)
            cats.append(obs_dict['vector_obs'])

        if self.message_head is not None:
            message = self.message_head(obs_dict['message'].float() / 255)
            cats.append(message)

        if len(cats) > 1:
            x = th.cat(cats, dim=1)

        return x


register_custom_encoder('nle_obs_vector_encoder', NLEMainEncoder)
