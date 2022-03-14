import matplotlib.pyplot

from utils import I2NR
from Dataloader import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm

res = 20
DATASETS = ["ACTOR", "TEXAS", "WISCONSIN",
            "CORNELL", 'AmazonComputers', 'Citeseer', 'Cora']
colors = ['lightcoral', 'orange', 'gold', 'cornflowerblue', 'violet', 'limegreen',
          'teal']



if __name__ == "__main__":
    hops = [i for i in range(1,res+1, 1)]
    fig, axes = plt.subplots(1, 1)
    axes = [axes]
    for color, dataset in tqdm(zip(colors, DATASETS)):
        data = load_data(dataset, 0)
        data.to(1)
        y = []
        for hop in hops:
            i2nr, _ = I2NR(data.edge_index, data.y, hop)
            y.append(i2nr)
        axes[0].plot(hops, y, label=dataset, c=color, linewidth=2)
    axes[0].set_ylabel("Information-to-Noise Ratio")
    axes[0].set_xlabel("Orders")
    axes[0].legend(loc='upper right')
    # axes[0].set_title("ratio")
    axes[0].grid(visible=True)
    axes[0].set_xticks([i for i in range(0, res+1, 2)])
    # plt.show()
    plt.savefig("graph-i2nr-ratio.pdf")


