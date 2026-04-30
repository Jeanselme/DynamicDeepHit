from ddh.losses import total_loss
from auton_survival.models.dsm.utilities import get_optimizer, _reshape_tensor_with_nans

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def train_ddh(model,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  alpha, beta, sigma,
			  n_iter = 10000, lr = 1e-3,
			  bs = 100, vbs = 500, cuda = False):

	optimizer = get_optimizer(model, lr)

	patience, old_loss = 0, np.inf
	nbatches = int(x_train.shape[0]/bs) + 1
	valbatches = int(x_valid.shape[0]/vbs) + 1

	for i in tqdm(range(n_iter)):
		model.train()
		for j in range(nbatches):
			xb = x_train[j*bs:(j+1)*bs]
			tb = t_train[j*bs:(j+1)*bs]
			eb = e_train[j*bs:(j+1)*bs]

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model,
							  xb,
							  tb,
							  eb,
 							  alpha, beta, sigma)
			loss.backward()
			optimizer.step()

		model.eval()
		valid_loss = 0
		for j in range(valbatches):
			xb = x_valid[j*bs:(j+1)*bs]
			tb = t_valid[j*bs:(j+1)*bs]
			eb = e_valid[j*bs:(j+1)*bs]

			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
			
			valid_loss += total_loss(model,
									xb,
									tb,
									eb,
									alpha, beta, sigma)

		valid_loss = valid_loss.item()
		if valid_loss < old_loss:
			patience = 0
			old_loss = valid_loss
			best_param = deepcopy(model.state_dict())
		else:
			if patience == 5:
				break
			else:
				patience += 1

	model.load_state_dict(best_param)
	return model

def create_nn(inputdim, outputdim, dropout = 0.6, layers = [100, 100], activation = 'ReLU', no_activation_last = False):
	modules = []
	if dropout > 0:
		modules.append(nn.Dropout(p = dropout))

	if activation == 'ReLU6':
		act = nn.ReLU6()
	elif activation == 'ReLU':
		act = nn.ReLU()
	elif activation == 'SeLU':
		act = nn.SELU()
	elif activation == 'Tanh':
		act = nn.Tanh()

	prevdim = inputdim
	for hidden in layers + [outputdim]:
		modules.append(nn.Linear(prevdim, hidden, bias = True))
		modules.append(act)
		prevdim = hidden

	if no_activation_last:
		modules = modules[:-1]

	return nn.Sequential(*modules)
