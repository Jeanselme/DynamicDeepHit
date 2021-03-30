import torch

def negative_log_likelihood(outcomes, cif, t, e):
    """
        Compute the log likelihood loss 
        This function is used to compute the survival loss
    """
    loss = 0

    # Censored
    loss += torch.sum(torch.log(1 - torch.sum(cif[e == 0], axis = 1) + 1e-10))

    # Uncensored
    for i, (ei, ti) in enumerate(zip(e, t)):
        if ei > 0:
            loss += torch.log(outcomes[ei-1][i, ti] + 1e-10)

    return - loss

def ranking_loss(outcomes, cif, t, e):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction
    """
    loss = 0
    for k in range(cif.shape[1]):
        for ci, ti in zip(cif[e-1 == k][:, k], t[e-1 == k]):
            # For all events: all patients that didn't experience event before
            # must have a lower risk for that cause
            loss += torch.sum(torch.DoubleTensor([torch.exp(- ci + torch.sum(oj[:ti+1])) for oj in outcomes[k][t > ti]]))

    return loss

def longitudinal_loss(longitudinal_prediction, x):
    """
        Penalize error in the longitudinal predictions
        This function is used to compute the error made by the RNN

        NB: In the paper, they seem to use different losses for continuous and categorical
        But this was not reflected in the code associated (therefore we compute MSE for all)

        NB: Original paper mentions possibility of different alphas for each risk
        But take same for all (for ranking loss)
    """
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1)

    # Select all predictions until the last observed
    predictions = torch.cat([longitudinal_prediction[i, :l - 1] for i, l in enumerate(length)], 0) 

    # Select all observations that can be predicted
    observations = torch.cat([x[i, 1:l] for i, l in enumerate(length)], 0) 
    return torch.nn.MSELoss(reduction = 'sum')(predictions, observations)

def total_loss(model, x, t, e, alpha, beta):
    longitudinal_prediction, outcomes = model(x)
    t, e = t.int(), e.int()
    
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    # Compute cumulative function from prediced outcomes
    # max_time = output_dim
    cif = torch.zeros((x.size(0), model.risks)).to(device)
    
    for k in range(model.risks):
        for i, (ti, oi) in enumerate(zip(t, outcomes[k])):
            cif[i, k] = torch.sum(oi[:ti+1])

    return longitudinal_loss(longitudinal_prediction, x) +\
              alpha * ranking_loss(outcomes, cif, t, e) +\
              beta * negative_log_likelihood(outcomes, cif, t, e)