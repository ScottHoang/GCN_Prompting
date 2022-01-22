parameters_dict = {
    'prompt_k': {
        'values' : [2, 3]
    },
    'prompt_lr': {
        'values': [0.4]
    },
    'lr': {
        'min': 1e-2,
        'max': 5e-2,
    },
    'prompt_pretrain_lr': {
        'min': 1e-3,
        'max': 1e-2
    },
    'prompt_pretrain_type': {
        'values' : ['edgeMask', 'edgeMask+contrastive', 'edgeMask+contrastive+attrMask']
    },
    'prompt_temp': {
        'values': [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    'prompt_distance_temp': {
        'values': [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    'prompt_neighbor_cutoff': {
        'values': [-1, 3, 5]},
    'prompt_layer': {
        'values': [16, 32],
    },
    'prompt_head':{
        'values' : ['GCN', 'SGC']
    },
    'prompt_aggr': {
        'values': ['concat', 'sum', 'edges'],
    },
    'epochs': {
        'values': [480, 960]
    },
    'prompt_continual': {
        'values': [True]
    },
    'prompt_type': {
        'values': ['micmap', 'micmip', 'macmip', 'class', 'classmicmip', 'classmicmap', 'classmacmip']
    },
    'alpha': {
        'values': [0.6, 0.7, 0.8, 0.9]
    },
    'dim_hidden' : {
      'values': [16,32,64,128]
    },
    'embedding_dropout' : {
        'min': 0.1,
        'max': 0.9
    }
}

def call():
    return parameters_dict