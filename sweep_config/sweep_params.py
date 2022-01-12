parameters_dict = {
    'prompt_k': {
        'values' : [1, 5, 10, 15] },
    'prompt_lr': {
        'values': [5e-1, 1e-1, 5e-2, 1e-2]},
    'lr': {
        'values': [5e-2, 1e-2, 5e-3, 1e-3]},
    'prompt_temp': {
        'values': [0.1, 0.5, 1.0, 5, 10]},
    'prompt_distance_temp': {
        'values': [0.1, 0.5, 1.0, 5, 10]},
    'prompt_neighbor_cutoff': {
        'values': [-1, 1, 2, 3, 4]},
    'prompt_layer': {
        'values': [1, 2],
    },
    'prompt_aggr': {
        'values': ['concat', 'sum', 'mean'],
    },
    'epochs': {
        'values': [120, 240, 480, 960, 1000]
    },
    'type_model': {
        'values': ['GCN', "VGAE"]
    },
    'prompt_continual': {
        'values': [True]
    },
    'prompt_w_org_features': {
        'values': [True]
    },
    'prompt_raw': {
        'values': [False]
    },
    'prompt_type': {
        'values': ['micmap', 'macmip', 'micmip']
    }




}