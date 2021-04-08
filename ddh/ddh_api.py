from DeepSurvivalMachines.dsm.dsm_api import DeepRecurrentSurvivalMachines
from dsm.utilities import _reshape_tensor_with_nans, _get_padded_features, _get_padded_targets
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
			DeepRecurrentSurvivalMachines ([type]): [description]
	"""

	def __init__(self, split = 50, layers_rnn = 1, typ = 'LSTM',
		hidden_long = 10, hidden_rnn = 10, hidden_att = 10, hidden_cs = 10,  
		alpha = 1, beta = 1, cuda = False):

		if isinstance(split, int):
			self.split = split
			self.split_time = None
		else:
			self.split = len(split) - 1
			self.split_time = split

		self.layers_rnn = layers_rnn
		self.typ = typ
		self.hidden_long = hidden_long
		self.hidden_rnn = hidden_rnn
		self.hidden_att = hidden_att
		self.hidden_cs = hidden_cs

		self.alpha = alpha
		self.beta = beta

		self.cuda = cuda
		self.fitted = False
		  
	def _gen_torch_model(self, inputdim, optimizer, risks):
		model = DynamicDeepHitTorch(inputdim, self.split, self.layers_rnn, 
								   self.hidden_long, self.hidden_rnn, 
								   self.hidden_att, self.hidden_cs,
								   self.typ, optimizer, risks).double()
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
		t, e = last(t), last(e)
		if not(val_data is None):
			val_data = val_data[0], last(val_data[1]), last(val_data[2])

		discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)
		processed_data = self._prepocess_training_data(x, discretized_t, e,
													vsize, val_data,
													random_state)
		x_train, t_train, e_train, x_val, t_val, e_val = processed_data
		inputdim = x_train.shape[-1]

		maxrisk = int(np.nanmax(e_train.cpu().numpy()))

		model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
		model = train_ddh(model,
					x_train, t_train, e_train,
					x_val, t_val, e_val,
					self.alpha, self.beta, 
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
			_, split_time = np.histogram(t, split - 1)
		try:
			t_discretized = np.digitize(t, split_time, right = True) - 1
		except:
			t_discretized = [np.digitize(t_, split_time, right = True) - 1 for t_ in t]
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
			t_val = torch.from_numpy(t_val).double()
			e_val = torch.from_numpy(e_val).double()

		return (x_train, t_train, e_train,				
				x_val, t_val, e_val)

	def compute_nll(self, x, t, e):
		if not self.fitted:
			raise Exception("The model has not been fitted yet. Please fit the " +
									"model using the `fit` method on some training data " +
									"before calling `_eval_nll`.")
		t, e = last(t), last(e)
		discretized_t, _ = self.discretize(t, self.split, self.split_time)
		processed_data = self._prepocess_training_data(x, discretized_t, e, 0, None, 0)
		_, _, _, x_val, t_val, e_val = processed_data
		return total_loss(self.torch_model, x_val, t_val, e_val, self.alpha, self.beta).item()

	def predict_survival(self, x, t, risk=1, all_step=False):
		l = [len(x_) for x_ in x]

		if all_step:
			new_x = []
			for x_, l_ in zip(x, l):
				new_x += [x_[:li + 1] for li in range(l_)]
			x = new_x

		x = self._prepocess_test_data(x)
		if not isinstance(t, list):
			t = [t]
		t, _ = self.discretize(t, self.split_time)

		if self.fitted:
			_, forecast = self.torch_model(x)
			forecast = forecast[int(risk) - 1]
			scores = []
			for t_ in t:
				scores.append(torch.sum(forecast[:, :t_+1], dim = 1).unsqueeze(1))
			return 1 - torch.cat(scores, dim = 1).detach().numpy()
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