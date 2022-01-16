import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def line_plot(axes, data, name):
    distances = data['distance']
    stats = data['stats']


    for i, (y_label, values) in enumerate(stats.items()):
        if y_label in ['train_loss', 'train_acc']: continue
        ax = axes[i-2]
        values = np.array(values)
        ax.plot(np.arange(values.shape[0]), values, label=name)
        ax.set_ylabel(y_label)
        # ax.legend()
        # ax.set_xlabel(x_label)
    axes[-2].plot(np.arange(len(distances['mean_distance'])), distances['mean_distance'], label=name)
    axes[-2].set_ylabel('mean_distance')
    axes[-1].plot(np.arange(len(distances['mean_i2nr'])), distances['mean_i2nr'], label=name)
    axes[-1].set_ylabel('mean_i2nr')
    # axes[-2].legend()
    # axes[0].legend()

def bar_plot(axes, data, name, pos):
    distances = data['distance']
    stats = data['stats']

    for i, (y_label, values) in enumerate(stats.items()):
        if y_label in ['train_loss', 'train_acc']: continue
        ax = axes[i-2]
        if 'los' in y_label:
            values = min(values)
            v2 = None
            text = f"{values:.4f}"
            ax.text(pos - 0.4, 0, text, color='black', fontweight='bold')
        else:
            if y_label == 'valid_acc':
                v2 = stats['test_acc'][np.argmax(values)] * 100
            else:
                v2 = stats['valid_acc'][np.argmax(values)] * 100
            values = max(values) * 100
            text = f"{values:.1f}|{v2:.1f}"
        ax.bar(pos, values, label=name)
        ax.text(pos-0.4, 0, text , color='black', fontweight='bold')
        ax.set_ylabel(y_label)
        # ax.legend()
        # ax.set_xlabel(x_label)
    axes[-2].bar(pos, max(distances['mean_distance']), label=name)
    axes[-2].set_ylabel('mean_distance')
    axes[-1].bar(pos, max(distances['mean_i2nr']), label=name)
    axes[-1].set_ylabel('mean_i2nr')
    # axes[-2].legend()
    axes[0].legend(bbox_to_anchor=(0., 0.9))


if __name__ == "__main__":
    _, src, dst, dataset = sys.argv

    dirs = os.listdir(src)

    fig, axes = plt.subplots(5, 2, figsize=(20, 18))
    cnt = 0
    for dir in dirs:
        path = os.path.join(src, dir, dataset)

        if os.path.isdir(path):
            sub_dirs = os.listdir(path)[-1]
            data = torch.load(os.path.join(path, sub_dirs, 'data.pth'))
            line_plot(axes[:,0], data, dir)
            bar_plot(axes[:, 1], data, dir, cnt)
            cnt+=1

    plt.savefig(dst)
    plt.show()








