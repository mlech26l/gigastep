import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker

a6000 = "\
 DEVICE &  1 & 2.3k & 12 hours  & 2.3k & 12 hours  \\\
 DEVICE &  8 & 26.0k & 64 minutes  & 21.4k & 78 minutes  \\\
 DEVICE &  32 & 106k & 16 minutes  & 88.3k & 19 minutes  \\\
 DEVICE &  128 & 421k & 4 minutes  & 350k & 5 minutes  \\\
 DEVICE &  512 & 1.7M & 58.9 seconds  & 818k & 2 minutes  \\\
 DEVICE &  2048 & 6.7M & 14.9 seconds  & 1.0M & 2 minutes  \\\
 DEVICE &  8192 & 26.8M & 3.7 seconds  & 1.0M & 2 minutes  \\\
 DEVICE &  32768 & 101M & 1.0 seconds  & 2.6G & 0.0 seconds  \\\
 DEVICE &  131056 & 250M & 0.4 seconds  & 10.5G & 0.0 seconds  \\\
 DEVICE &  524224 & 289M & 0.3 seconds  & 41.9G & 0.0 seconds  \\\
"
a6000_2 = "\
  DEVICE &  1 & 2.4k & 11 hours  & 2.4k & 12 hours  \\\
 DEVICE &  8 & 26.4k & 63 minutes  & 22.2k & 75 minutes  \\\
 DEVICE &  32 & 100k & 17 minutes  & 88.7k & 19 minutes  \\\
 DEVICE &  128 & 422k & 4 minutes  & 353k & 5 minutes  \\\
 DEVICE &  512 & 1.7M & 58.7 seconds  & 1.2M & 1 minutes  \\\
 DEVICE &  2048 & 6.7M & 14.9 seconds  & 1.2M & 1 minutes  \\\
 DEVICE &  8192 & 26.4M & 3.8 seconds  & 1.4M & 1 minutes  \\\
"
a100 = "\
 DEVICE &  1 & 2.5k & 11 hours  & 2.4k & 11 hours  \\\
 DEVICE &  8 & 26.6k & 63 minutes  & 19.6k & 85 minutes  \\\
 DEVICE &  32 & 94.3k & 18 minutes  & 87.6k & 19 minutes  \\\
 DEVICE &  128 & 408k & 4 minutes  & 354k & 5 minutes  \\\
 DEVICE &  512 & 1.7M & 58.7 seconds  & 979k & 2 minutes  \\\
 DEVICE &  2048 & 6.7M & 15.0 seconds  & 1.3M & 1 minutes  \\\
 DEVICE &  8192 & 26.8M & 3.7 seconds  & 1.4M & 1 minutes  \\\
 DEVICE &  32768 & 107M & 0.9 seconds  & 2.6G & 0.0 seconds  \\\
 DEVICE &  131056 & 314M & 0.3 seconds  & 10.5G & 0.0 seconds  \\\
 DEVICE &  524224 & 380M & 0.3 seconds  & 41.9G & 0.0 seconds  \\\
"
rtx3090 = "\
 DEVICE &  1 & 3.1k & 9 hours  & 3.1k & 9 hours  \\\
 DEVICE &  8 & 33.7k & 49 minutes  & 28.2k & 59 minutes  \\\
 DEVICE &  32 & 138k & 12 minutes  & 112k & 15 minutes  \\\
 DEVICE &  128 & 540k & 3 minutes  & 463k & 4 minutes  \\\
 DEVICE &  512 & 2.2M & 46.1 seconds  & 955k & 2 minutes  \\\
 DEVICE &  2048 & 8.7M & 11.6 seconds  & 1.2M & 1 minutes  \\\
 DEVICE &  8192 & 35.0M & 2.9 seconds  & 1.2M & 1 minutes  \\\
 DEVICE &  32768 & 133M & 0.8 seconds  & 2.6G & 0.0 seconds  \\\
 DEVICE &  131056 & 280M & 0.4 seconds  & 10.5G & 0.0 seconds  \\\
 DEVICE &  524224 & 323M & 0.3 seconds  & 41.9G & 0.0 seconds  \\\
"
rtx2080 = "\
 DEVICE &  1 & 5.5k & 5 hours  & 5.7k & 5 hours  \\\
 DEVICE &  8 & 64.5k & 26 minutes  & 54.5k & 31 minutes  \\\
 DEVICE &  32 & 259k & 6 minutes  & 211k & 8 minutes  \\\
 DEVICE &  128 & 969k & 2 minutes  & 491k & 3 minutes  \\\
 DEVICE &  512 & 4.0M & 25.1 seconds  & 677k & 2 minutes  \\\
 DEVICE &  2048 & 16.5M & 6.1 seconds  & 746k & 2 minutes  \\\
 DEVICE &  8192 & 64.9M & 1.5 seconds  & 763k & 2 minutes  \\\
 DEVICE &  32768 & 178M & 0.6 seconds  & 2.6G & 0.0 seconds  \\\
 DEVICE &  131056 & 200M & 0.5 seconds  & 10.5G & 0.0 seconds  \\\
 DEVICE &  524224 & 211M & 0.5 seconds  & 41.9G & 0.0 seconds  \\\
 "


def prase_number(text):
    factor = 1
    if "k" in text:
        factor = 1000
        text = text.replace("k", "")
    elif "M" in text:
        factor = 1000000
        text = text.replace("M", "")
    elif "G" in text:
        factor = 1000000 * 1000
        text = text.replace("G", "")
    return float(text) * factor


def prase_str(text):
    text = text.split("\\")
    x, y1, y2 = [], [], []
    for line in text:
        line = line.split("&")
        if len(line) < 6:
            continue
        x.append(int(line[1]))
        y1.append(prase_number(line[2]))
        y2.append(prase_number(line[4]))

    return x, y1, y2


def nice_str(x):
    if x >= 10000000:
        return f"{x/1000000:.0f}M"
    elif x >= 1000000:
        return f"{x/1000000:.1f}M"
    elif x >= 10000:
        return f"{x/1000:.0f}k"
    elif x >= 1000:
        return f"{x/1000:.1f}k"
    else:
        return f"{x:.0f}"


def nice_bs(x):
    if x > 1024:
        return f"{x//1024}k"
    else:
        return f"{x}"


sns.set(style="whitegrid", palette="muted", font_scale=1.0)


# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

colors = sns.color_palette()
for i, (gpu, text) in enumerate(
    [("RTX 2080Ti", rtx2080), ("RTX 3090", rtx3090), ("A6000", a6000), ("A100", a100)]
):
    x, y1, y2 = prase_str(text)
    x2 = x[0:7]
    y2 = y2[0:7]
    ax1.scatter(x, y1, color=colors[i])
    ax2.scatter(x2, y2, color=colors[i])
    ax1.plot(x, y1, label=gpu, color=colors[i])
    ax2.plot(x2, y2, label=gpu, color=colors[i])
ax1.set_xscale("log")
ax2.set_xscale("log")

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")

# Set titles and labels
ax1.set_title("Vector observation")
ax2.set_title("RGB observation")
ax1.set_xlabel("Batch size")
ax1.set_ylabel("Steps per second")
ax2.set_xlabel("Batch size")
ax2.set_ylabel("Steps per second")
ax1.set_xticks(
    [1, 8, 32, 128, 512, 2048, 8192, 8192 * 4, 8192 * 4 * 4, 8192 * 4 * 4 * 4]
)
ax2.set_xticks([1, 8, 32, 128, 512, 2048, 8192])
ax1.xaxis.set_major_formatter(lambda x, pos: nice_bs(x))
ax2.xaxis.set_major_formatter(lambda x, pos: nice_bs(x))
ax1.yaxis.set_major_formatter(lambda x, pos: nice_str(x))
ax2.yaxis.set_major_formatter(lambda x, pos: nice_str(x))

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.3)
fig.tight_layout()
# Display the figure
fig.savefig("batch_size.png", bbox_inches="tight")
fig.savefig("batch_size.pdf", bbox_inches="tight")