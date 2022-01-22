parameters_dict = {
    'prompt_k': {
        'values' : [3, 4, 5] },
    'prompt_lr': {
        'min': 1e-2,
        'max': 5e-1,
    },
    'lr': {
        'min': 1e-2,
        'max': 5e-2,
    },
    'prompt_pretrain_lr': {
        'min': 1e-2,
        'max': 5e-2
    },
    'prompt_pretrain_type': {
        'values' : ['edgeMask', 'edgeMask+contrastive',]
    },
    'prompt_temp': {
        'values': [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    ,
    'prompt_distance_temp': {
        'min':0.1,
        'max':1.0,
    },
    'prompt_neighbor_cutoff': {
        'values': [-1, 3]},
    'prompt_layer': {
        'values': [8, 16],
    },
    'prompt_head':{
        'values' : ['GCN', "SGC"]
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
        'values': ['micmap', 'micmip', 'macmip', 'classmicmip', 'classmicmap', 'classmacmip']
    },
    'alpha': {
        'values': [0.6, 0.7, 0.8, 0.9]
    },
    'dim_hidden' : {
      'values': [16,32,64]
    },
    'embedding_dropout' : {
        'min': 0.1,
        'max': 0.6
    }
}
def call():
    return parameters_dict