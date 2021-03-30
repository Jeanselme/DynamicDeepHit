import torch.nn as nn
import torch
import numpy as np

class DynamicDeepHitTorch(nn.Module):

	def __init__(self, input_dim, output_dim, layers_rnn,
				hidden_long, hidden_rnn, hidden_att, hidden_cs,
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
		self.longitudinal = nn.Sequential(
							nn.Linear(hidden_rnn, hidden_long, bias=True),
							nn.ReLU6(),
							nn.Linear(hidden_long, hidden_long, bias=True),
							nn.ReLU6(),
							nn.Linear(hidden_long, input_dim, bias=True),
						)

		# Attention mechanism
		self.attention = nn.Sequential(
							nn.Linear(input_dim + hidden_rnn, hidden_att, bias=True),
							nn.ReLU6(),
							nn.Linear(hidden_att, hidden_att, bias=True),
							nn.ReLU6(),
							nn.Linear(hidden_att, 1, bias=True)
						)
		self.attention_soft = nn.Softmax(1) # On temporal dimension

		# Cause specific network
		self.cause_specific = []
		for r in range(self.risks):
			self.cause_specific.append(
						nn.Sequential(
							nn.Linear(hidden_rnn, hidden_cs, bias=True), 
							nn.ReLU6(),
							nn.Linear(hidden_cs, hidden_cs, bias=True),
							nn.ReLU6(),
							nn.Linear(hidden_cs, output_dim, bias=True)
						).double())

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
		last_observations_idx = ~inputmask.sum(axis = 1)
		last_observations = torch.cat([x[i, j].repeat(1, x.size(1), 1) for i, j in enumerate(last_observations_idx)], 0) 

		## Concatenate all previous with new to measure attention
		concatenation = torch.cat([hidden, last_observations], -1)

		## Compute attention and normalize
		attention = self.attention(concatenation).squeeze()
		attention[inputmask] = 0
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
