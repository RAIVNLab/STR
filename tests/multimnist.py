import sys, os
sys.path.append(os.path.abspath('.'))

from args import args
from data import MultiMNIST
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

args.data = "/usr/data"
args.num_train_examples = 200000
args.num_val_examples = 50000
args.num_concat = 4
args.workers = 16

mnist = MultiMNIST(args)

label_counts = defaultdict(int)
for i in tqdm.tqdm(range(len(mnist.train_loader.dataset)), ascii=True):
    _, label = mnist.train_loader.dataset[i]
    label_counts[label.item()] += 1


fig, (ax1, ax2) = plt.subplots(2)

sns.kdeplot(label_counts, ax=ax1)

plt.plot(*zip(*sorted(label_counts.items())))
plt.savefig("tests/images/multimnist.pdf", bbox_inches="tight")
