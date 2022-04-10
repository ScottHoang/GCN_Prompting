import collections
import os
import os.path as osp
import pandas as pd
import sys

if __name__ == "__main__":
    _, path, save_dir, output_name = sys.argv

    #
    dataframes = {}
    for dataset in os.listdir(path):
        path1 = osp.join(path, dataset)
        data = collections.defaultdict(dict)
        for model in os.listdir(path1):
            path2 = osp.join(path1, model)
            for pretrain_task in os.listdir(path2):
                path3 = osp.join(path2, pretrain_task)
                f = osp.join(path3, os.listdir(path3)[-1])
                results = {}#collections.defaultdict(dict)
                results['model'] = model
                results['dataset'] = dataset
                results['task'] = pretrain_task
                with open(f) as file:
                    for line in file:
                        pass
                    if 'final' in line: # last line
                        line = [l.strip(" ") for l in line.strip().split(',')[2::] if l != '' and 'acc' in l or 'roc_auc' in l]
                        for l in line:
                            name, number = l.split(' ')
                            name = name.split("_")[0]
                            acc, std = number.split(':')
                            results[name+"_acc"] = float(acc)
                            results[name+"_std"] = float(std)
                    else: # something has gone wrong, typically OOM
                        names = ['train', 'valid', 'test']
                        for n in names:
                            results[n + "_acc"] = -1.0 #float(acc)
                            results[n  + "_std"] = -1.0 #float(std)
                    data[pretrain_task][model] = results
        dataframes[dataset] = data
    # generate 2d data-frame (to be import to google sheet later as csv)
    pd_df = pd.DataFrame()
    for dataset in dataframes.keys():
        for task in dataframes[dataset].keys():
            _pd_df = pd.DataFrame.from_dict(dataframes[dataset][task], orient='index')
            pd_df = pd.concat([pd_df, _pd_df], ignore_index=True)
    os.makedirs(save_dir, exist_ok=True)
    pd_df.to_csv(osp.join(save_dir, output_name), index=False)