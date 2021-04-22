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
		self.longitudinal = create_nn(hidden_rnn, input_dim, **long_param)

		# Attention mechanism
		self.attention = create_nn(input_dim + hidden_rnn, 1, **att_param)
		self.attention_soft = nn.Softmax(1) # On temporal dimension

		# Cause specific network
		self.cause_specific = []
		for r in range(self.risks):
			self.cause_specific.append(create_nn(hidden_rnn, output_dim, **cs_param))
		self.cause_specific = nn.ModuleList(self.cause_specific)

		# Probability
		self.soft = nn.Softmax(dim = -1) # On all observed output

	def forward(self, x):
		"""
			The forward function that is called when data is passed through DynamicDeepHit.
		"""
		# RNN representation - Nan values for not observed data
		x = x.clone()
		inputmask = torch.isnan(x[:, :, 0])
		x[inputmask] = 0
		hidden, _ = self.embedding(x)		
		
		# Longitudinal modelling
		longitudinal_prediction = self.longitudinal(hidden)

		# Attention using last observation to predict weight of all previously observed
		## Extract last observation (the one used for predictions)
		last_observations_idx = (~inputmask).sum(axis = 1)
		last_observations = torch.cat([x[i, j - 1].repeat(1, x.size(1), 1) for i, j in enumerate(last_observations_idx)], 0) 

		## Concatenate all previous with new to measure attention
		concatenation = torch.cat([hidden, last_observations], -1)

		## Compute attention and normalize
		attention = self.attention(concatenation).squeeze(-1)
		attention[inputmask] = -1e10 # Want soft max to be zero as values not observed
		attention = self.attention_soft(attention)

		# Risk networks
		outcomes = []
		attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
		hidden_attentive = torch.sum(attention * hidden, axis = 1)
		for cs_nn in self.cause_specific:
			outcomes.append(cs_nn(hidden_attentive))

		# Soft max for probability distribution
		outcomes = torch.cat(outcomes, dim = 1)
		outcomes = self.soft(outcomes)

		outcomes = [outcomes[:, i * self.output_dim : (i+1) * self.output_dim] for i in range(self.risks)]
		return longitudinal_prediction, outcomes
