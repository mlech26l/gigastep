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
            n_team1_wins_bk = n_team1_wins
            n_team1_wins = int(
                n_team2_wins
            )  # HACK: to compensate for the mismatch bug in network.py and cross_eval.py; int(n_team1_wins)
            n_team2_wins = int(
                n_team1_wins_bk
            )  # HACK: to compensate for the mismatch bug in network.py and cross_eval.py; int(n_team2_wins)
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


def add_M_and_remove_zeros(name):
    new_name = int(name[:-2])
    return f"{new_name}M"


def plot_wins(filename, env_name, results, ckpts):
    # sns.set("paper", "whitegrid", "dark", font_scale=1.5, rc={"lines.linewidth": 2})
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 3))

    # sort by name (zfilled)
    ckpts = sorted(ckpts)
    results = [results[c] for c in ckpts]
    # for t, c in enumerate(ckpts):
    #     print(f"Ckpt {t}: {c}")
    win_rates = [100 * r[0] / r[1] for r in results]
    x = np.arange(len(win_rates))
    ax.set_title(env_name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [add_M_and_remove_zeros(c) for c in ckpts])
    # ax.xaxis.set_major_formatter(lambda x, pos: f"{x}M")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:0.0f}%")
    ax.scatter(x, win_rates)
    ax.plot(x, win_rates)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def main():
    keys = ["identical_5_vs_5_det"]
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/pdf", exist_ok=True)
    results_by_env, table_by_env = load_csv("cross_eval_results_self_play.csv")
    # results_by_env, table_by_env = load_csv("cross_eval_results.csv")
    for env, results in results_by_env.items():
        ckpts = list(results.keys())
        ckpts_1 = [c for c in ckpts if c.endswith("_1")]
        ckpts_2 = [c for c in ckpts if c.endswith("_2")]
        plot_wins(f"plots/{env}_1.png", env, results, ckpts_1)
        plot_wins(f"plots/{env}_2.png", env, results, ckpts_2)
        plot_wins(f"plots/pdf/{env}_1.pdf", env, results, ckpts_1)
        plot_wins(f"plots/pdf/{env}_2.pdf", env, results, ckpts_2)

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
        plt.figure(figsize=(5, 4))
        heatmap = plt.imshow(win_rates, cmap="YlGnBu")
        # hide grid
        plt.grid(False)

        is_self_play = ["_2" in c for c in all_ckpts]
        is_self_play = not np.any(is_self_play)

        if is_self_play:
            # Remove _1 from the name (there is no _2)
            ckpt_ticks = [add_M_and_remove_zeros(c) for c in all_ckpts]
        else:
            ckpt_ticks = [add_M_and_remove_zeros(c) + c[-2:] for c in all_ckpts]

        # Set the x- and y-tick labels
        plt.xticks(np.arange(len(all_ckpts)), ckpt_ticks, rotation=90)
        plt.yticks(np.arange(len(all_ckpts)), ckpt_ticks)
        # rotate x ticks 90 degrees

        # Add a colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Win Rate (%)")

        plt.xlabel("Opponent")
        plt.ylabel("Player")
        plt.title(f"Win Rate for {env}")
        plt.tight_layout()
        plt.savefig(f"plots/{env}_heatmap.png")
        plt.savefig(f"plots/pdf/{env}_heatmap.pdf")
        plt.close()


if __name__ == "__main__":
    main()