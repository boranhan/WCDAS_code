CosLoss = {
    'loss_params': {'loss':'CosLoss'},
}

WCDAS_ImageNetLT = {
    'loss_params': {'loss':'WCDAS', 'gamma':-1.},
}

WCDAS_iNaturalist2018 = {
    'loss_params': {'loss':'WCDAS', 'gamma':1., 's_trainable':False},
}

WCDAS_CIFARLT = {
    'loss_params': {'loss':'WCDAS', 'gamma':0},
}

