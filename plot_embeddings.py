import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE as skTSNE
from sklearn.decomposition import  PCA
import imageio
import sys
from tqdm import tqdm

colors = {
    0: 'coral',
    1: 'seagreen',
    2: 'b',
    3: 'm',
    4: 'r',
    5: 'gold',
    6: 'indigo',
    7: 'slategrey',
    8: 'aquamarine',
    9: 'dimgrey',
    10: 'teal'
}

def read_files(paths):
    coordinates_lists = []
    prompts = []
    labels = None
    lr = []
    test_acc = []
    edge_index = None
    tsne = skTSNE(n_components=2, learning_rate='auto', init='pca')
    pca = PCA(n_components=50)
    for p in paths:
        file = torch.load(p)
        embs = file['final_embs']
        embs = embs.detach().cpu().numpy()
        embs = pca.fit_transform(embs)
        particles = tsne.fit_transform(embs).transpose()
        coordinates_lists.append(particles)
        prompts.append(file['prompts'].cpu().numpy())
        test_acc.append(file['results']['test_acc'])
        lr.append(file['lr'])
        if labels is None:
            labels = file['labels'].cpu().numpy()
        if edge_index is None:
            edge_index = file['edges'].cpu().numpy()
    return labels, coordinates_lists, prompts, lr, test_acc, edge_index


def generate_scatter_plots(labels, coordinates_lists, prompts, lr, test_acc, edge_index,
                           n_frames=20, marker_size=10, bg_color='#95A4AD'):
    global colors
    filenames = []
    xmin_avg, xmax_avg, ymin_avg, ymax_avg = [], [],[],[]
    for index in np.arange(0, len(coordinates_lists)):
        # get current and next coordinates
        x = coordinates_lists[index][0]
        y = coordinates_lists[index][1]
        xmin_avg.append(min(x))
        xmax_avg.append(max(x))
        ymin_avg.append(min(y))
        ymax_avg.append(max(y))
    offset = 5
    xmin = np.average(xmin_avg) - offset
    xmax = np.average(xmax_avg) + offset
    ymin = np.average(ymin_avg) - offset
    ymax = np.average(ymax_avg) + offset

    for index in np.arange(0, len(coordinates_lists) - 1):
        # get current and next coordinates
        x = coordinates_lists[index][0]
        y = coordinates_lists[index][1]
        x1 = coordinates_lists[index + 1][0]
        y1 = coordinates_lists[index + 1][1]
        prompt = prompts[index]

        while len(x) < len(x1):
            diff = len(x1) - len(x)
            x = x + x[:diff]
            y = y + y[:diff]
        while len(x1) < len(x):
            diff = len(x) - len(x1)
            x1 = x1 + x1[:diff]
            y1 = y1 + y1[:diff]
        # calculate paths
        x_path = np.array(x1) - np.array(x)
        y_path = np.array(y1) - np.array(y)
        for i in np.arange(0, n_frames + 1):
            # calculate current position
            x_temp = (x + (x_path / n_frames) * i)
            y_temp = (y + (y_path / n_frames) * i)
            x_temp = np.clip(x_temp, xmin, xmax)
            y_temp = np.clip(y_temp, ymin, ymax)
            # plot
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
            ax.set_facecolor(bg_color)
            for src, tgt in edge_index.transpose():
                plt.plot(x_temp[[src,tgt]], y_temp[[src,tgt]], linewidth=1, alpha=0.25, c='gainsboro')
            upscale = [1 for _ in range(x_temp.shape[0])]
            for src, tgts in enumerate(prompt):
                if tgts is None:
                    break
                else:
                    for tgt in tgts:
                        if src == tgt: continue
                        else:
                            upscale[tgt] += 1
            scale = np.array([marker_size*2**np.log(n) for n in upscale])
            for g in np.unique(labels):
                ix = np.where(labels==g)
                plt.scatter(x_temp[ix], y_temp[ix], c=colors[g], s=scale[ix], label=g)
            plt.title(f'training at epoch: {i+index*n_frames}, lr: {lr[index]:.3f}, acc: {test_acc[index]*100:.2f}')
            plt.legend()
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            # remove spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # grid
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
            ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
            # build file name and append to list of file names
            filename = f'images/frame_{index}_{i}.png'
            filenames.append(filename)
            if (i == n_frames):
                for i in range(5):
                    filenames.append(filename)
            # save img
            plt.savefig(filename, dpi=96, facecolor=bg_color)
            plt.close()
    return filenames

def create_gif(filenames, gif_name):
    print('creating gif\n')
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('gif complete\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('done')



if __name__ == "__main__":
    _, dst, dataset = sys.argv
    root = 'embeddings'
    if not os.path.isdir('images'):
        os.makedirs('images')
    if not os.path.isdir(dst):
        os.makedirs(dst)
    totalpaths = []
    for paths in os.listdir(root):
        path = os.path.join(root, paths)
        if dataset in os.listdir(path):
            path = os.path.join(path, dataset)
            directory = os.listdir(path)[-1]
            files = [os.path.join(path, directory, f) for f in os.listdir(os.path.join(path, directory))]
            totalpaths.append(files)

    for paths in tqdm(totalpaths):
        gif_name = os.path.join(dst, paths[0].split('/')[1])
        labels, coordinates, prompts, lrs, test_acc, edge_index= read_files(paths)
        filenames = generate_scatter_plots(labels, coordinates, prompts, lrs, test_acc, edge_index)
        create_gif(filenames, gif_name)

    os.rmdir('images')
