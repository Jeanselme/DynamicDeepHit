from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses import total_loss
from dsm.utilities import get_optimizer, _reshape_tensor_with_nans

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def train_ddh(model,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  alpha, beta,
			  n_iter = 10000, lr = 1e-3,
			  bs = 100, cuda = False):

	optimizer = get_optimizer(model, lr)

	patience, old_loss = 0, np.inf
	nbatches = int(x_train.shape[0]/bs) + 1
	valbatches = int(x_valid.shape[0]/bs) + 1

	for i in tqdm(range(n_iter)):
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
 							  alpha, beta)
			loss.backward()
			optimizer.step()

		valid_loss = 0
		for j in range(valbatches):
			xb = x_valid[j*bs:(j+1)*bs]
			tb = t_valid[j*bs:(j+1)*bs]
			eb = e_valid[j*bs:(j+1)*bs]

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
			
			valid_loss += total_loss(model,
									xb,
									tb,
									eb,
									alpha, beta)

		valid_loss = valid_loss.item()
		if valid_loss < old_loss:
			patience = 0
			old_loss = valid_loss
			best_param = deepcopy(model.state_dict())
		else:
			if patience == 2:
				break
			else:
				patience += 1

	model.load_state_dict(best_param)
	return model
