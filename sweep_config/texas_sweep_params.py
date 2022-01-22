parameters_dict = {
    'prompt_k': {
        'values' : [3, 4, 5] },
    'prompt_lr': {
        'values': [0.45]
    },
    'lr': {
        'min': 1e-3,
        'max': 5e-3,
    },
    'prompt_pretrain_lr': {
        'min': 5e-3,
        'max': 5e-2
    },
    'prompt_pretrain_type': {
        'values' : ['edgeMask', 'edgeMask+contrastive']
    },
    'prompt_temp': {
        'min':0.1,
        'max':1.0}
    ,
    'prompt_distance_temp': {
        'min':0.1,
        'max':5.0,
    },
    'prompt_neighbor_cutoff': {
        'values': [-1]},
    'prompt_layer': {
        'values': [16, 32],
    },
    'prompt_head':{
        'values' : ['GCN', "SGC"]
    },
    'prompt_aggr': {
        'values': ['concat', 'edges'],
    },
    'epochs': {
        'values': [480, 960, 1000]
    },
    'prompt_continual': {
        'values': [True]
    },
    'prompt_type': {
        'values': ['micmap', 'micmip', 'macmip', 'classmicmip', 'classmicmap', 'classmacmip']
    },
    'alpha': {
        'values': [0.6]
    },
    'dim_hidden' : {
      'values': [32,64,128]
    },
    'embedding_dropout' : {
        'values': [0.6]
    }
}
def call():
    return parameters_dict
