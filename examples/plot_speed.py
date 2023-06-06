import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker

a6000 = "\
    DEVICE &   1 & 2.3k & 12 hours  & 2.3k & 12 hours  \\\
 DEVICE &  8 & 26.3k & 63 minutes  & 21.5k & 78 minutes  \\\
 DEVICE &  32 & 103k & 16 minutes  & 87.9k & 19 minutes  \\\
 DEVICE &  128 & 423k & 4 minutes  & 355k & 5 minutes  \\\
 DEVICE &  512 & 1.7M & 59.2 seconds  & 817k & 2 minutes  \\\
 DEVICE &  2048 & 6.7M & 15.0 seconds  & 1.0M & 2 minutes  \\\
 DEVICE &  8192 & 26.6M & 3.8 seconds  & 1.0M & 2 minutes  \\\
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
  DEVICE &  1 & 2.5k & 11 hours  & 2.3k & 12 hours  \\\
 DEVICE &  8 & 24.1k & 69 minutes  & 19.7k & 85 minutes  \\\
 DEVICE &  32 & 96.5k & 17 minutes  & 80.1k & 21 minutes  \\\
 DEVICE &  128 & 374k & 4 minutes  & 350k & 5 minutes  \\\
 DEVICE &  512 & 1.7M & 58.9 seconds  & 983k & 2 minutes  \\\
 DEVICE &  2048 & 6.1M & 16.4 seconds  & 1.3M & 1 minutes  \\\
 DEVICE &  8192 & 27.3M & 3.7 seconds  & 1.4M & 1 minutes  \\\
"
rtx3090 = "\
  DEVICE &  1 & 3.0k & 9 hours  & 3.1k & 9 hours  \\\
 DEVICE &  8 & 33.2k & 50 minutes  & 28.2k & 59 minutes  \\\
 DEVICE &  32 & 132k & 13 minutes  & 112k & 15 minutes  \\\
 DEVICE &  128 & 534k & 3 minutes  & 455k & 4 minutes  \\\
 DEVICE &  512 & 2.1M & 46.7 seconds  & 955k & 2 minutes  \\\
 DEVICE &  2048 & 8.5M & 11.8 seconds  & 1.2M & 1 minutes  \\\
 DEVICE &  8192 & 34.3M & 2.9 seconds  & 1.2M & 1 minutes  \\\
"
rtx2080 = "\
 DEVICE &  1 & 5.6k & 5 hours  & 5.7k & 5 hours  \\\
 DEVICE &  8 & 65.0k & 26 minutes  & 54.4k & 31 minutes  \\\
 DEVICE &  32 & 260k & 6 minutes  & 217k & 8 minutes  \\\
 DEVICE &  128 & 1.0M & 2 minutes  & 493k & 3 minutes  \\\
 DEVICE &  512 & 4.1M & 24.4 seconds  & 677k & 2 minutes  \\\
 DEVICE &  2048 & 16.6M & 6.0 seconds  & 745k & 2 minutes  \\\
 DEVICE &  8192 & 65.1M & 1.5 seconds  & 763k & 2 minutes  \\\
 "


def prase_number(text):
    factor = 1
    if "k" in text:
        factor = 1000
        text = text.replace("k", "")
    elif "M" in text:
        factor = 1000000
        text = text.replace("M", "")
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
    if x > 1000000:
        return f"{x/1000000:.0f}M"
    elif x > 1000:
        return f"{x/1000:.0f}k"
    else:
        return f"{x:.1f}"


sns.set(style="whitegrid", palette="muted", font_scale=1.2)


# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

colors = sns.color_palette()
for i, (gpu, text) in enumerate(
    [("RTX 2080Ti", rtx2080), ("RTX 3090", rtx3090), ("A6000", a6000), ("A100", a100)]
):
    x, y1, y2 = prase_str(text)
    ax1.scatter(x, y1, color=colors[i])
    ax2.scatter(x, y2, color=colors[i])
    ax1.plot(x, y1, label=gpu, color=colors[i])
    ax2.plot(x, y2, label=gpu, color=colors[i])
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
ax1.set_xticks([1, 8, 32, 128, 512, 2048, 8192])
ax2.set_xticks([1, 8, 32, 128, 512, 2048, 8192])
ax1.xaxis.set_major_formatter(lambda x, pos: f"{x}")
ax2.xaxis.set_major_formatter(lambda x, pos: f"{x}")
ax1.yaxis.set_major_formatter(lambda x, pos: nice_str(x))
ax2.yaxis.set_major_formatter(lambda x, pos: nice_str(x))

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.3)
fig.tight_layout()
# Display the figure
fig.savefig("batch_size.png", bbox_inches="tight")
fig.savefig("batch_size.pdf", bbox_inches="tight")