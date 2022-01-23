parameters_dict = {
    'prompt_k': {
        'values' : [2, 3, 4, 5] },
    'prompt_lr': {
        'min': 1e-1,
        'max': 5e-1,
    },
    'lr': {
        'min': 1e-3,
        'max': 5e-2,
    },
    'prompt_pretrain_lr': {
        'min': 1e-3,
        'max': 5e-2
    },
    'prompt_pretrain_type': {
        'values' : ['edgeMask', 'contrastive', 'attrMask', 'edgeMask+contrastive', 'edgeMask+attrMask', 'edgeMask+contrastive+attrMask']
    },
    'prompt_temp': {
        'min':0.1,
        'max':5.0}
    ,
    'prompt_distance_temp': {
        'min':0.1,
        'max':5.0,
    },
    'prompt_neighbor_cutoff': {
        'values': [-1, 3, 5]},
    'prompt_layer': {
        'values': [3, 8, 16, 32],
    },
    'prompt_head':{
        'values' : ['GCN', "SGC"]
    },
    'prompt_aggr': {
        'values': ['concat', 'sum', 'mean', 'edges'],
    },
    'epochs': {
        'values': [120, 240, 480, 960, 1000]
    },
    'prompt_continual': {
        'values': [True, False]
    },
    'prompt_type': {
        'values': ['micmap', 'micmip', 'macmip', 'macmap']
    },
    'alpha': {
        'values': [0.6, 0.7, 0.8, 0.9]
    },
    'dim_hidden' : {
      'values': [16,32,64,128,256]
    },
    'prompt_dim_hidden' : {
        'values': [16,32,64,128,256]
    },
    'embedding_dropout' : {
        'min': 0.3,
        'max': 0.6
    }
}