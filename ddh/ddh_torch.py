import torch.nn as nn
import torch
import numpy as np
from .utils import create_nn

class DynamicDeepHitTorch(nn.Module):

	def __init__(self, input_dim, output_dim, layers_rnn,
				hidden_rnn, long_param = {}, att_param = {}, cs_param = {},
				typ = 'LSTM', optimizer = 'Adam', risks = 1):
		super(DynamicDeepHitTorch, self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.optimizer = optimizer
		self.risks = risks
		self.typ = typ

		# RNN model for longitudinal data
		if self.typ == 'LSTM':
			self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True)
		if self.typ == 'RNN':
			self.embedding = nn.RNN(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True,
								nonlinearity='relu')
		if self.typ == 'GRU':
			self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn,
								bias=False, batch_first=True)

		# Longitudinal network
		self.longitudinal = create_nn(hidden_rnn, input_dim, no_activation_last = True, **long_param)

		# Attention mechanism
		self.attention = create_nn(input_dim + hidden_rnn, 1, no_activation_last = True, **att_param)
		self.attention_soft = nn.Softmax(1) # On temporal dimension

		# Cause specific network
		self.cause_specific = []
		for r in range(self.risks):
			self.cause_specific.append(create_nn(input_dim + hidden_rnn, output_dim, no_activation_last = True, **cs_param))
		self.cause_specific = nn.ModuleList(self.cause_specific)

		# Probability
		self.soft = nn.Softmax(dim = -1) # On all observed output

	def forward(self, x):
		"""
			The forward function that is called when data is passed through DynamicDeepHit.
		"""
		if x.is_cuda:
			device = x.get_device()
		else:
			device = torch.device("cpu")

		# RNN representation - Nan values for not observed data
		x = x.clone()
		inputmask = torch.isnan(x[:, :, 0])
		x[inputmask] = 0
		hidden, _ = self.embedding(x)		
		
		# Longitudinal modelling
		longitudinal_prediction = self.longitudinal(hidden)

		# Attention using last observation to predict weight of all previously observed
		## Extract last observation (the one used for predictions)
		last_observations = ((~inputmask).sum(axis = 1) - 1)
		last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
		index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

		last = index == last_observations_idx
		x_last = x[last]

		## Concatenate all previous with new to measure attention
		concatenation = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1)

		## Compute attention and normalize
		attention = self.attention(concatenation).squeeze(-1)
		attention[index >= last_observations_idx] = -1e10 # Want soft max to be zero as values not observed
		attention[last_observations > 0] = self.attention_soft(attention[last_observations > 0]) # Weight previous observation
		attention[last_observations == 0] = 0 # No context for only one observation

		# Risk networks
		# The original paper is not clear on how the last observation is
		# combined with the temporal sum, other code was concatenating them
		outcomes = []
		attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
		hidden_attentive = torch.sum(attention * hidden, axis = 1)
		hidden_attentive = torch.cat([hidden_attentive, x_last], 1)
		for cs_nn in self.cause_specific:
			outcomes.append(cs_nn(hidden_attentive))

		# Soft max for probability distribution
		outcomes = torch.cat(outcomes, dim = 1)
		outcomes = self.soft(outcomes)

		outcomes = [outcomes[:, i * self.output_dim : (i+1) * self.output_dim] for i in range(self.risks)]
		return longitudinal_prediction, outcomes
