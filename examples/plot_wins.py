import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv(path):
    results_by_env = {}
    table_by_env = {}
    with open(path) as f:
        for line in f:
            (
                env,
                mode,
                n_team1_wins,
                n_team2_wins,
                n_games,
                ckpt1,
                ckpt2,
            ) = line.strip().split(",")

            if env == "env":
                continue
            if env not in results_by_env:
                results_by_env[env] = {}
                table_by_env[env] = {}
            results_by_ckpt = results_by_env[env]

            n_games = int(n_games)
            n_team1_wins = int(n_team1_wins)
            n_team2_wins = int(n_team2_wins)
            draws = n_games - n_team1_wins - n_team2_wins
            assert draws >= 0

            ckpt1 = ckpt1.split("/")[-1][: -len("000000")]
            ckpt2 = ckpt2.split("/")[-1][: -len("000000")]
            ckpt1 = ckpt1.zfill(3)
            ckpt2 = ckpt2.zfill(3)

            if mode[0] == "1":
                ckpt1 = ckpt1 + "_1"
            else:
                ckpt1 = ckpt1 + "_2"
            if mode[1] == "1":
                ckpt2 = ckpt2 + "_1"
            else:
                ckpt2 = ckpt2 + "_2"

            key1 = f"{ckpt1}_vs_{ckpt2}"
            key2 = f"{ckpt2}_vs_{ckpt1}"
            assert key1 not in results_by_ckpt
            assert key2 not in results_by_ckpt
            table_by_env[env][key2] = n_team2_wins / n_games
            table_by_env[env][key1] = n_team1_wins / n_games

            if ckpt1 not in results_by_ckpt:
                results_by_ckpt[ckpt1] = [0, 0]
            if ckpt2 not in results_by_ckpt:
                results_by_ckpt[ckpt2] = [0, 0]

            results_by_ckpt[ckpt1][0] += n_team1_wins
            results_by_ckpt[ckpt1][1] += n_games
            results_by_ckpt[ckpt2][0] += n_team2_wins
            results_by_ckpt[ckpt2][1] += n_games
            results_by_env[env] = results_by_ckpt
    return results_by_env, table_by_env


def plot_wins(filename, env_name, results, ckpts):
    # sns.set("paper", "whitegrid", "dark", font_scale=1.5, rc={"lines.linewidth": 2})
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()

    # sort by name (zfilled)
    ckpts = sorted(ckpts)
    results = [results[c] for c in ckpts]
    # for t, c in enumerate(ckpts):
    #     print(f"Ckpt {t}: {c}")
    win_rates = [r[0] / r[1] for r in results]
    x = np.arange(2, len(win_rates) * 2 + 2, 2)
    ax.set_title(env_name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(2, len(win_rates) * 2 + 2, 2))
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x}M")
    ax.scatter(x, win_rates)
    ax.plot(x, win_rates)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def main():
    keys = ["identical_5_vs_5_det"]
    os.makedirs("plots", exist_ok=True)
    results_by_env, table_by_env = load_csv("cross_eval_results.csv")
    for env, results in results_by_env.items():
        ckpts = list(results.keys())
        ckpts_1 = [c for c in ckpts if c.endswith("_1")]
        ckpts_2 = [c for c in ckpts if c.endswith("_2")]
        plot_wins(f"plots/{env}_1.png", env, results, ckpts_1)
        plot_wins(f"plots/{env}_2.png", env, results, ckpts_2)

    # for env, table in table_by_env.items():
    # for env in keys:
    for env in table_by_env.keys():
        table = table_by_env[env]
        print("Plotting heatmap for", env)
        all_ckpts = sorted(list(results_by_env[env].keys()))

        win_rates = np.zeros((len(all_ckpts), len(all_ckpts)))
        for i1, p1 in enumerate(all_ckpts):
            for i2, p2 in enumerate(all_ckpts):
                keystr = f"{p1}_vs_{p2}"
                win_rates[i1, i2] = table.get(keystr, 0)
                if keystr not in table:
                    print(f"Missing {keystr}")

        win_rates = 100 * win_rates
        heatmap = plt.imshow(win_rates, cmap="YlGnBu")
        # hide grid
        plt.grid(False)

        # Set the x- and y-tick labels
        plt.xticks(np.arange(len(all_ckpts)), all_ckpts, rotation=90)
        plt.yticks(np.arange(len(all_ckpts)), all_ckpts)
        # rotate x ticks 90 degrees

        # Add annotations
        # for i in range(len(all_ckpts)):
        #     for j in range(len(all_ckpts)):
        #         plt.text(
        #             j,
        #             i,
        #             f"{win_rates[i, j]:0.1f}",
        #             ha="center",
        #             va="center",
        #             color="black",
        #         )

        # Add a colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Win Rate")

        plt.xlabel("Opponent")
        plt.ylabel("Player")
        plt.title("Win Rate Heatmap")
        plt.tight_layout()
        plt.savefig(f"plots/{env}_heatmap.png")
        plt.close()


if __name__ == "__main__":
    main()