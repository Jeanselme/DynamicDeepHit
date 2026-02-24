from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from auton_survival.models.dsm.utilities import _reshape_tensor_with_nans, _get_padded_features, _get_padded_targets
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.utils import train_ddh
from ddh.losses import total_loss
import numpy as np
import torch

last = lambda t: np.array([t_[-1] for t_ in t], dtype = float)

class DynamicDeepHit(DeepRecurrentSurvivalMachines):
	"""
		This implementation considers that the last event happen at the same time for each patient
		The CIF is therefore simplified

		Args:
			DeepRecurrentSurvivalMachines
	"""

	def __init__(self, split = 50, layers_rnn = 1, hidden_rnn = 10, typ = 'LSTM',
		long_param = {}, att_param = {}, cs_param = {},
		alpha = 0.1, beta = 0.1, sigma = 0.1, cuda = False):

		if isinstance(split, int):
			self.split = split
			self.split_time = None
		else:
			self.split = len(split) - 1
			self.split_time = split

		self.layers_rnn = layers_rnn
		self.hidden_rnn = hidden_rnn
		self.typ = typ
		
		self.long_param = long_param
		self.att_param = att_param
		self.cs_param = cs_param

		self.alpha = alpha
		self.beta = beta
		self.sigma = sigma

		self.cuda = cuda
		self.fitted = False
		  
	def _gen_torch_model(self, inputdim, optimizer, risks):
		model = DynamicDeepHitTorch(inputdim, self.split, self.layers_rnn, self.hidden_rnn,
				long_param = self.long_param, att_param = self.att_param, cs_param = self.cs_param,
				typ = self.typ, optimizer = optimizer, risks = risks).double()
		if self.cuda:
			model = model.cuda()
		return model

	def cpu(self):
		self.cuda = False
		if self.torch_model:
			self.torch_model = self.torch_model.cpu()
		return self

	def fit(self, x, t, e, vsize = 0.15, val_data = None,
		  iters = 1, learning_rate = 1e-3, batch_size = 100,
		  optimizer = "Adam", random_state = 100):
		discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)
		processed_data = self._prepocess_training_data(x, last(discretized_t), last(e),
													vsize, val_data,
													random_state)
		x_train, t_train, e_train, x_val, t_val, e_val = processed_data
		inputdim = x_train.shape[-1]

		maxrisk = int(np.nanmax(e_train.cpu().numpy()))

		model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
		model = train_ddh(model,
					x_train, t_train, e_train,
					x_val, t_val, e_val,
					self.alpha, self.beta, self.sigma, 
					n_iter = iters,
					lr = learning_rate,
					bs = batch_size,
					cuda = self.cuda)

		self.torch_model = model.eval()
		self.fitted = True
		self.cpu()

		return self  

	def discretize(self, t, split, split_time = None):
		"""
			Discretize the survival horizon

			Args:
				t (List of Array): Time of events
				split (int): Number of bins
				split_time (List, optional): List of bins (must be same length than split). Defaults to None.

			Returns:
				List of Array: Disretized events time
		"""
		if split_time is None:
			_, split_time = np.histogram(np.concatenate(t), split - 1)
		t_discretized = np.array([np.digitize(t_, split_time, right = True) - 1 for t_ in t], dtype = object)
		return t_discretized, split_time

	def _prepocess_test_data(self, x):
		data = torch.from_numpy(_get_padded_features(x))
		if self.cuda:
			data = data.cuda()
		return data

	def _prepocess_training_data(self, x, t, e, vsize, val_data, random_state):
		"""RNNs require different preprocessing for variable length sequences"""

		idx = list(range(x.shape[0]))
		np.random.seed(random_state)
		np.random.shuffle(idx)

		x = _get_padded_features(x)
		x_train, t_train, e_train = x[idx], t[idx], e[idx]

		x_train = torch.from_numpy(x_train).double()
		t_train = torch.from_numpy(t_train).double()
		e_train = torch.from_numpy(e_train).double()

		if val_data is None:

			vsize = int(vsize*x_train.shape[0])

			x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]
            
			x_train = x_train[:-vsize]
			t_train = t_train[:-vsize]
			e_train = e_train[:-vsize]

		else:

			x_val, t_val, e_val = val_data

			x_val = _get_padded_features(x_val)
			t_val, _ = self.discretize(t_val, self.split, self.split_time)

			x_val = torch.from_numpy(x_val).double()
			t_val = torch.from_numpy(last(t_val)).double()
			e_val = torch.from_numpy(last(e_val)).double()

		return (x_train, t_train, e_train,				
				x_val, t_val, e_val)

	def compute_nll(self, x, t, e):
		if not self.fitted:
			raise Exception("The model has not been fitted yet. Please fit the " +
									"model using the `fit` method on some training data " +
									"before calling `_eval_nll`.")
		discretized_t, _ = self.discretize(t, self.split, self.split_time)
		processed_data = self._prepocess_training_data(x, last(discretized_t), last(e), 0, None, 0)
		_, _, _, x_val, t_val, e_val = processed_data
		return total_loss(self.torch_model, x_val, t_val, e_val, 0, 1, 1).item()

	def predict_survival(self, x, t, risk=1, all_step=False, fill_from_last=False, bs=100):
		"""Predict survival function for the given data.
		Args:
			x (List of Array): Input data, where each element is a sequence of features for a patient.
			t (Array): Time points at which to evaluate the survival function.
			risk (int, optional): Risk index to predict. Defaults to 1.
			all_step (bool, optional): If True, returns survival function for all time steps otherwise only last. 
				Defaults to False.
			fill_from_last (bool, optional): If True, return all steps from last time point to first. 
				Defaults to False (index 0 is the first observed time, otherwise last observed point). 
				Ignored if all_step is False.
			bs (int, optional): Batch size for prediction. Defaults to 100.
		Returns:
			Array(point, time_eval, (time_step)): Predicted survival function for each patient at the specified time points."""
		l = [len(x_) for x_ in x]

		if all_step:
			new_x = []
			for x_, l_ in zip(x, l):
				new_x += [x_[:li + 1] for li in range(l_)]
			x = new_x

		if not isinstance(t, list):
			t = [t]
		t = self.discretize([t], self.split, self.split_time)[0][0]

		if self.fitted:
			batches = int(len(x)/bs) + 1
			scores = {t_: [] for t_ in t}
			for j in range(batches):
				xb = self._prepocess_test_data(x[j*bs:(j+1)*bs])
				_, f = self.torch_model(xb)
				for t_ in t:
					scores[t_].append(torch.cumsum(f[int(risk) - 1], dim = 1)[:, t_].unsqueeze(1).detach().numpy())
			predictions = 1 - np.concatenate([np.concatenate(scores[t_], axis = 0) for t_ in t], axis = 1)
			if all_step:
				results = np.full((len(l), len(t), max(l)), float('nan'))
				start = 0
				for i, li in enumerate(l):
					if fill_from_last:
						# Reverse to have index 0 containing last observed time
						results[i, :, :li] = predictions[start:start+li][::-1].T
					else:
						results[i, :, :li] = predictions[start:start+li].T
					start += li
				return results
			else:
				return predictions
		else:
			raise Exception("The model has not been fitted yet. Please fit the " +
							"model using the `fit` method on some training data " +
							"before calling `predict_survival`.")

	def predict_risk(self, x, t, **args):
		if self.fitted:
			return 1-self.predict_survival(x, t, **args)
		else:
			raise Exception("The model has not been fitted yet. Please fit the " +
							"model using the `fit` method on some training data " +
							"before calling `predict_risk`.")