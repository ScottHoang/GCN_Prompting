import argparse
MODELS = ['GCN', 'GAT', 'SGC', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN', 'VGAE']

class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description='Constrained learing')

        parser.add_argument("--dataset", type=str, default="Cora", required=False,
                            help="The input dataset.",
                            choices=['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv',
                                     'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhoto',
                                     'TEXAS', 'WISCONSIN', 'ACTOR', 'CORNELL'])
        # build up the common parameter
        parser.add_argument('--random_seed', type=int, default=100)
        parser.add_argument('--N_exp', type=int, default=100)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="run in cuda mode")
        parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
        parser.add_argument('--log_file_name', type=str, default='time_and_memory.log')

        parser.add_argument('--compare_model', type=int, default=0,
                            help="0: test tricks, 1: test models")

        parser.add_argument('--type_model', type=str, default="GCN",
                            choices=MODELS)
        parser.add_argument('--type_trick', type=str, default="None")
        parser.add_argument('--layer_agg', type=str, default='concat',
                            choices=['concat', 'maxpool', 'attention', 'mean'],
                            help='aggregation function for skip connections')

        parser.add_argument('--num_layers', type=int, default=64)
        parser.add_argument("--epochs", type=int, default=1000,
                            help="number of training the one shot model")
        parser.add_argument('--patience', type=int, default=100,
                            help="patience step for early stopping")  # 5e-4
        parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
        parser.add_argument("--dropout", type=float, default=0.6,
                            help="dropout for GCN")
        parser.add_argument('--embedding_dropout', type=float, default=0.6,
                            help='dropout for embeddings')
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help="weight decay")  # 5e-4
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--transductive', type=bool, default=True,
                            help='transductive or inductive setting')
        parser.add_argument('--activation', type=str, default="relu", required=False)

        # task [node, edge, prompt]
        parser.add_argument('--task', type=str, default='node')
        # edge task specific parameters for data prep
        parser.add_argument('--use-splitted', action='store_true',
                            help='use the pre-splitted train/test data,\
                             if False, then make a random division')
        parser.add_argument('--data-split-num', type=str, default='10',
                            help='If use-splitted is true, choose one of splitted data')
        parser.add_argument('--test-ratio', type=float, default=0.3,
                            help='ratio of test links')
        parser.add_argument('--val-ratio', type=float, default=0.1,
                            help='ratio of validation links. If using the splitted data from SEAL,\
                             it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
        parser.add_argument('--practical-neg-sample', type=bool, default=False,
                            help='only see the train positive edges when sampling negative')
        # setups in peparing the training set
        parser.add_argument('--observe-val-and-injection', action='store_true',
                            help='whether to contain the validation set in the observed graph and apply injection trick')

        # prepare initial node attributes for those graphs do not have
        parser.add_argument('--init-attribute', action='store_true',
                            help='initial attribute for graphs without node attributes\
                            , options: n2v, one_hot, spc, ones, zeros, None')
        # batch size for edge
        parser.add_argument('--batch-size', type=int, default=1024)
        # prompt parameters
        parser.add_argument('--prompt-record', action='store_true')
        parser.add_argument('--prompt-k', type=int, default=5)
        parser.add_argument('--prompt-raw', action='store_true')
        parser.add_argument('--prompt-continual', action='store_true')
        parser.add_argument('--prompt-aggr', type=str, default='concat', help='concat|sum|max|mean')
        parser.add_argument('--prompt-head', type=str, default='mlp', help='mlp|gnn')
        parser.add_argument('--prompt-head-lr', type=float, default=5e-2)
        parser.add_argument('--prompt-layer', type=int, default=2)
        parser.add_argument('--prompt-opt', type=str, default='head', help='both|head')
        parser.add_argument('--prompt-type', type=str, default='bfr', help='bfs|mad')
        parser.add_argument('--prompt-lr', type=float, default=1e-3)
        parser.add_argument('--prompt-w-org-features', action='store_true')
        parser.add_argument('--prompt-save-embs', action='store_true')
        parser.add_argument('--prompt-get-mad', action='store_true')
        parser.add_argument('--prompt-mode', type=str,  default='',
                            help='setting all related mode; format: '
                                 '{type_model}-{prompt_head}.{num_layer}-{prompt_layer}.{prompt-k}-{prompt-w-org-features}-{prompt-aggr}.{prompt_type}.{prompt-raw}-{prompt_continual}')
        ###
        # Hyperparameters for specific model, such as GCNII, EdgeDropping, APPNNP, PairNorm
        parser.add_argument('--alpha', type=float, default=0.1,
                            help="residual weight for input embedding")
        parser.add_argument('--lamda', type=float, default=0.5,
                            help="used in identity_mapping and GCNII")
        parser.add_argument('--weight_decay1', type=float, default=0.01, help='weight decay in some models')
        parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay in some models')
        parser.add_argument('--type_norm', type=str, default="None")
        parser.add_argument('--adj_dropout', type=float, default=0.5,
                            help="dropout rate in APPNP")  # 5e-4
        parser.add_argument('--edge_dropout', type=float, default=0.2,
                            help="dropout rate in EdgeDrop")  # 5e-4

        parser.add_argument('--node_norm_type', type=str, default="n", choices=['n', 'v', 'm', 'srv', 'pr'])
        parser.add_argument('--skip_weight', type=float, default=None)
        parser.add_argument('--num_groups', type=int, default=None)
        parser.add_argument('--has_residual_MLP', type=bool, default=False)

        # Hyperparameters for random dropout
        parser.add_argument('--graph_dropout', type=float, default=0.2,
                            help="graph dropout rate (for dropout tricks)")  # 5e-4
        parser.add_argument('--layerwise_dropout', action='store_true', default=False)

        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)

        return args

    ## setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout = 0.6  # 0.5
            args.lr = 0.005  # 0.005
            args.weight_decay = 5e-4
            # args.epochs = 1000
            # args.patience = 100
            # args.dim_hidden = 64
            args.activation = 'relu'
            # edge task specific
            args.use_splitted = False
            # args.practical_neg_sample = True
            args.observe_val_and_injection = False
            args.init_attribute = False

            # args.N_exp = 100

        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.dropout = 0.5
            args.lr = 0.01
            args.weight_decay = 5e-4
            # args.epochs = 1000
            # args.patience = 100
            # args.dim_hidden = 256
            args.activation = 'relu'
            # edge task specific
            # args.use_splitted = False
            # # args.practical_neg_sample = True
            # args.observe_val_and_injection = False
            # args.init_attribute = False

        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6

            args.dropout = 0.7
            args.lr = 0.01
            args.lamda = 0.6
            args.weight_decay = 5e-4
            # args.epochs = 1000
            # args.patience = 100
            # args.dim_hidden = 256
            args.activation = 'relu'

            args.res_alpha = 0.2
            # edge task specific
            args.use_splitted = False
            # # args.practical_neg_sample = True
            # args.observe_val_and_injection = False
            # args.init_attribute = False

        elif args.dataset == 'ogbn-arxiv':
            args.num_feats = 128
            args.num_classes = 40
            args.dropout = 0.1
            args.lr = 0.005
            args.weight_decay = 0.
            # args.epochs = 1000
            # args.patience = 200
            # args.dim_hidden = 256


        # ==============================================
        # ========== below are other datasets ==========

        elif args.dataset == 'CoauthorPhysics':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 8415
            args.num_classes = 5

            args.dropout = 0.8
            args.lr = 0.005
            args.weight_decay = 0.



        elif args.dataset == 'CoauthorCS':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 6805
            args.num_classes = 15

            args.dropout = 0.8
            args.lr = 0.005
            args.weight_decay = 0.




        elif args.dataset == 'TEXAS':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.6
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9




        elif args.dataset == 'WISCONSIN':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.6
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9


        elif args.dataset == 'CORNELL':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9


        elif args.dataset == 'ACTOR':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 932
            args.num_classes = 5

            args.dropout = 0.
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9

        elif args.dataset == 'AmazonComputers':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 767
            args.num_classes = 10

            args.dropout = 0.5
            args.lr = 0.005
            args.weight_decay = 5e-5

        elif args.dataset == 'AmazonPhoto':
            # args.epochs = 1000
            # args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 745
            args.num_classes = 8

            args.dropout = 0.5
            args.lr = 0.005
            args.weight_decay = 5e-4


        if args.prompt_mode != "":
            #'{type_model}-{prompt_head}.{num_layer}-{prompt_layer}.{prompt-k}-{prompt-w-org-features}-{prompt-aggr}.{prompt_type}.{raw}-{contiual}')
            settings = args.prompt_mode.split('.')
            assert len(settings) == 5
            assert len(settings[0].split('-')) == 2
            type_model, prompt_head = settings[0].split('-')
            assert type_model in MODELS
            args.type_model = type_model
            args.prompt_head = prompt_head
            assert  len(settings[1].split('-')) == 2
            args.num_layers, args.prompt_layer = [int(i) for i in settings[1].split('-')]
            if len(settings[2].split('-')) == 2:
                args.prompt_k, args.prompt_aggr = settings[2].split('-')
            elif len(settings[2].split('-')) == 3:
                args.prompt_w_org_features = True
                args.prompt_k, _, args.prompt_aggr = settings[2].split('-')
            else:
                raise ValueError
            args.prompt_k = int(args.prompt_k)
            args.prompt_type = settings[3]

            assert len(settings[4].split('-')) == 2
            raw, continual = settings[4].split('-')
            args.prompt_raw = raw == 'r'
            args.prompt_continual = continual == 'c'
            return args
