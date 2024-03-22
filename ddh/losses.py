import torch

def negative_log_likelihood(outcomes, cif, t, e):
    """
        Compute the log likelihood loss 
        This function is used to compute the survival loss
    """
    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        # Censored cif
        censored_cif += cif[k][e == 0][torch.arange((e == 0).sum()), t[e == 0]]
        # Uncensored
        selection = e == (k + 1)
        loss += torch.sum(torch.log(ok[selection][torch.arange((selection).sum()), t[selection]] + 1e-10))

    # Censored loss
    loss += torch.sum(torch.log(1 - censored_cif + 1e-10))
    return - loss / len(outcomes)

def ranking_loss(cif, t, e, sigma):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction
    """
    loss = 0
    # Data ordered by time
    for k, cifk in enumerate(cif):
        for ci, ti in zip(cifk[e-1 == k], t[e-1 == k]):
            # For all events: all patients that didn't experience event before
            # must have a lower risk for that cause
            if torch.sum(t > ti) > 0:
                # TODO: When data are sorted in time -> wan we make it even faster ?
                loss += torch.mean(torch.exp((cifk[t > ti][torch.arange((t > ti).sum()), ti] - ci[ti])) / sigma)

    return loss / len(cif)

def longitudinal_loss(longitudinal_prediction, x):
    """
        Penalize error in the longitudinal predictions
        This function is used to compute the error made by the RNN

        NB: In the paper, they seem to use different losses for continuous and categorical
        But this was not reflected in the code associated (therefore we compute MSE for all)

        NB: Original paper mentions possibility of different alphas for each risk
        But take same for all (for ranking loss)
    """
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation

    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x[observation_mask])

def total_loss(model, x, t, e, alpha, beta, sigma):
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    return (1 - alpha - beta) * longitudinal_loss(longitudinal_prediction, x) +\
              alpha * ranking_loss(cif, t, e, sigma) +\
              beta * negative_log_likelihood(outcomes, cif, t, e)