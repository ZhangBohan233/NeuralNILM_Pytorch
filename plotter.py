import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nilmtk.losses as losses

RANGES = {
    "ukdale": {
        "fridge": (250, 2000),
        # "kettle": (7500, 9500),
        "kettle": (8500, 9100),
        # "microwave": (68400, 69500),
        "microwave": (68450, 68850),
        "dish washer": (83500, 85100),
        "washing machine": (121800, 122500)
    }
}

GRAPH_NAMES = {
    "SGN": "SGN",
    "AttentionCNN": "Attention CNN",
    "DM": "U-Net-DM",
    "DMGated": "GT-NILM$^\mathrm{GM}$"
}

APP_NAMES = {
    "kettle": "Kettle",
    "fridge": "Refrigerator",
    "microwave": "Microwave oven",
    "dish washer": "Dish washer",
    "washing machine": "Washing machine"
}

# CSFONT = {'fontname': 'Times New Roman'}


LINE_STYLES = [
    'dotted',
    'dashed',
    'dashdot',
    'solid'
]


def load_app_gate(app_name, path):
    truth_name = app_name + "_truth"
    pred_name = app_name + "_pred"
    gate_name = app_name + "_gate"
    ungated_name = app_name + "_ungated"

    df_orig = pd.read_csv(path, index_col=0)

    df = df_orig.loc[:, ("mains", truth_name, pred_name, gate_name, ungated_name)]
    df.columns = ["mains", "truth", "pred", "gate", "ungated"]

    return df


def plot_app_gate(app, path: str, rng):
    df = load_app_gate(app, path)
    # rng = RANGES["ukda"][app]
    plt.figure(figsize=(12, 8))

    # plt.subplots(2, 1, figsize=(12, 6))
    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()
    pred = df.loc[:, "pred"].to_numpy()
    gate = df.loc[:, "gate"].to_numpy() * 2000
    ung = df.loc[:, "ungated"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.subplot(2, 1, 1)
    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Mains")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=1, label="Predicted")

    plt.subplot(2, 1, 2)
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    # plt.plot(x_val, gate[rng[0]: rng[1]], linewidth=1, label="Gate signal")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2, label="Predicted")
    plt.plot(x_val, ung[rng[0]: rng[1]], linewidth=1, label="Original")

    plt.xlabel("Samples")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")

    save_path = path[path.rfind("/")].replace('csv', 'png')
    plt.savefig(f"./csv_ft/figures/{save_path}", bbox_inches='tight')

    plt.show()


def plot_app_transfer(app, path: str, rng):
    df = load_app_gate(app, path)
    # rng = RANGES["ukda"][app]
    plt.figure(figsize=(12, 6))

    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()
    pred = df.loc[:, "pred"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.xlim((0, rng[1] - rng[0]))

    # plt.figure(figsize=(12, 6))
    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Aggregated power")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")
    plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2, label="GT-NILM",
             color='#9467bd')

    plt.xlabel("Time interval index")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = path[path.rfind("/") + 1:].replace('csv', 'png')
    plt.savefig(f"./csv_ft/figures/tsf_{save_path}", bbox_inches='tight')

    plt.show()


def load_app(app_name, clf_dict: dict):
    truth_name = app_name + "_truth"
    pred_name = app_name + "_pred"

    ix = None
    df_list = []
    for clf, csv_file in clf_dict.items():
        subframe = pd.read_csv("results/" + csv_file, index_col=0)
        if ix is None:
            ix = subframe.index
        else:
            ix = ix.intersection(subframe.index)
        df_list.append((clf, subframe))

    df = pd.DataFrame()
    for clf, subframe in df_list:
        subframe = subframe.loc[ix]

        if len(df) == 0:
            df = subframe.loc[:, ("mains", truth_name, pred_name)]
            df.columns = ["mains", "truth", clf]
            df.index = ix
        else:
            df[clf] = subframe[pred_name]

    return df


def plot_app(app, df, clf_names, ds, score=True, cut=None):
    rng = RANGES["ukdale"][app]
    plt.figure(figsize=(12, 6))
    plt.title(APP_NAMES[app], fontweight='bold')

    if cut is not None:
        plt.ylim((0, cut))

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)
    plt.xlim((0, rng[1] - rng[0]))

    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=2, label="Aggregated power")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")

    line_style_index = 0
    for clf in clf_names:
        pred = df.loc[:, clf].to_numpy()
        # x_val = np.arange(rng[1] - rng[0]) * 6

        if score:
            acc = losses.accuracy(app, truth, pred)
            f1 = losses.f1score(app, truth, pred)
            pre = losses.precision(app, truth, pred)
            recall = losses.recall(app, truth, pred)
            mae = losses.mae(app, truth, pred)
            sae = losses.sae(app, truth, pred)
            print(f"{clf}: {acc:.4f}, F1: {f1:.3f}, MAE: {mae:.2f}, SAE: {sae:.2f}, "
                  f"Precision: {pre:.3f}, Recall: {recall:.3f}, ")

        plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=2,
                 label=GRAPH_NAMES[clf], linestyle=LINE_STYLES[line_style_index])
        line_style_index += 1

    plt.xlabel("Time interval index")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(f"./figures/{ds}_{app}.png", bbox_inches='tight')

    plt.show()


def draw_direct():
    df_fridge = load_app("fridge", {
        "AttentionCNN": "ukdale-fridge-AttentionCNN.csv",
        "SGN": "ukdale-fridge-SGN.csv",
        "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
        "DMGated": "gated-ukdale-fridge-DM_GATE2.csv"
    })

    print(df_fridge.head())
    plot_app("fridge", df_fridge, ["AttentionCNN", "SGN", "DM", "DMGated"],
             'ukdale', score=False, cut=600)

    # df_dish = load_app("dish washer", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-dish washer-SGN.csv",
    #     "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    #     "DMGated": "gated-ukdale-dish washer-DM_GATE2.csv"
    # })
    #
    # print(df_dish.head())
    # plot_app("dish washer",
    #          df_dish,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'ukdale', score=False)
    #
    # df_wash = load_app("washing machine", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-washing machine-SGN.csv",
    #     "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    #     "DMGated": "gated-ukdale-washing machine-DM_GATE2.csv"
    # })
    #
    # print(df_wash.head())
    # plot_app("washing machine",
    #          df_wash,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'ukdale', score=False)
    #
    # df_kettle = load_app("kettle", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-kettle-SGN.csv",
    #     "DM": "ukdale-kettle+fridge-DM_GATE2.csv",
    #     "DMGated": "gated-ukdale-kettle+microwave-DM_GATE2.csv",
    # })
    #
    # print(df_kettle.head())
    # plot_app("kettle",
    #          df_kettle,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'ukdale', score=False)

    df_microwave = load_app("microwave", {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-microwave-SGN.csv",
        "DM": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
        "DMGated": "gated-ukdale-kettle+microwave-DM_GATE2.csv",
    })

    print(df_microwave.head())
    plot_app("microwave",
             df_microwave,
             ["AttentionCNN", "SGN", "DM", "DMGated"],
             'ukdale', score=False)


def draw_redd():
    df_fridge = load_app("fridge", {
        "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
        "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
        "DM": "redd-fridge+washing machine+dish washer-DM_GATE2.csv",
        "DMGated": "gated-redd-fridge-DM_GATE2.csv"
    })

    print(df_fridge.head())
    plot_app("fridge", df_fridge, ["AttentionCNN", "SGN", "DM", "DMGated"], 'redd', score=True)

    # df_dish = load_app("dish washer", {
    #     "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
    #     "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
    #     "DM": "redd-fridge+washing machine+dish washer-DM_GATE2.csv",
    #     "DMGated": "gated-redd-dish washer+washing machine-DM_GATE2.csv"
    # })
    #
    # print(df_dish.head())
    # plot_app("dish washer",
    #          df_dish,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'redd', score=True)

    # df_wash = load_app("washing machine", {
    #     "AttentionCNN": "redd-dish washer+washing machine+fridge-AttentionCNN.csv",
    #     "SGN": "redd-dish washer+washing machine+fridge-SGN.csv",
    #     "DM": "redd-fridge+washing machine+dish washer-DM_GATE2.csv",
    #     "DMGated": "gated-redd-dish washer+washing machine-DM_GATE2.csv"
    # })
    #
    # print(df_wash.head())
    # plot_app("washing machine",
    #          df_wash,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'redd', score=True)

    # df_microwave = load_app("microwave", {
    #     "AttentionCNN": "redd-microwave-AttentionCNN.csv",
    #     "SGN": "redd-microwave-SGN.csv",
    #     "DM": "redd-microwave-DM_GATE2.csv",
    #     "DMGated": "gated-redd-microwave-DM_GATE2.csv"
    # })
    #
    # print(df_microwave.head())
    # plot_app("microwave",
    #          df_microwave,
    #          ["AttentionCNN", "SGN", "DM", "DMGated"],
    #          'redd',
    #          score=True)


def draw_transfer():
    # plot_app_gate("microwave",
    #               "./csv_ft/microwave_3_GATE=True_DM=True.csv",
    #               (27500, 28500))
    plot_app_transfer("washing machine",
                      "./csv_ft/washing machine_1_GATE=True_DM=True.csv",
                      (69150, 70800))
    plot_app_transfer("dish washer",
                      "./csv_ft/dish washer_1_GATE=True_DM=True.csv",
                      (9500, 11000)
                      # (34500, 36000)
                      )


def draw_bar():
    df = pd.read_csv("./metrics/time.csv")
    print(df.head())
    # mapp = {'attn': 'AttentionCNN',
    #         'sgn': 'SGN',
    #         'dm': 'DM',
    #         'dm-gate': 'DM-with-Filter'}
    res = {}

    for app in APP_NAMES:
        model = df[df['app'] == app]
        times = list(model['total_time'].values)
        res[app] = times

    # for model in mapp:
    #     apps = df[df['model'] == model]
    #     name = model
    #     times = list(apps['total_time'].values)
    #     res[mapp[name]] = times

    print(res)

    order = ['Refrigerator', 'Dish washer', 'Washing machine', 'Microwave oven', 'Kettle']

    plt.figure(figsize=(12, 6))

    # colors = ['#e9824d', '#d6d717', '#92cad1', '#868686', '#79ccb3']
    # colors = ['#4e669e', '#699b87', '#86ac51', '#fcc602', '#f0df72']
    colors = ['#92C9C0', '#F8F4BD', '#C1BED5', '#EF9388', '#868686']
    hatches = ['/', '//', '--', 'x', '+']

    x = np.arange(4)
    pos = -1 / 3
    i = 0
    for name, values in res.items():
        xs = x + pos
        plt.bar(xs, values, 1 / 6, align='center', color=colors[i], hatch=hatches[i])
        pos += 1 / 6
        i += 1

    plt.xticks(x, ['Attention CNN', 'SGN', 'U-Net-DM', 'GT-NILM$^\mathrm{GM}$'])
    plt.xlabel("Method")
    plt.ylabel("Training time (seconds)")
    plt.legend([APP_NAMES[app] for app in APP_NAMES])
    plt.tight_layout()
    plt.savefig(f"./figures/time_cmp.png", bbox_inches='tight')
    plt.show()


def plot_loss(path, plt_index, title=None):
    df = pd.read_csv(path)
    df = df.dropna()

    # plt.subplot(1, 2, plt_index)

    x = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values

    # plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    min_index = np.argmin(val_loss)
    print(val_loss, min_index)
    plt.axvline(x=min_index, color='orange')

    # if plt_index == 1:
    plt.ylabel('Loss')
    plt.xlabel('Epoch index')

    if title is not None:
        plt.title(title)

    # plt.legend()


def plot_losses():
    plt.figure(figsize=(4, 3))
    # fig.tight_layout(pad=5.0)
    # plot_loss('./training_logs/dish_ukdale_dm/loss.csv', 1, 'U-Net-DM')
    plot_loss('./training_logs/dish_ukdale_dm_g2/loss.csv', 1)
    plt.tight_layout()

    # plt.subplots_adjust(wspace=0.25, bottom=0.15)

    plt.savefig(f"./figures/loss.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    font = {'family': 'Times New Roman',
            'weight': 'normal'}
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22}
    matplotlib.rc('font', **font)
    # matplotlib.rcParams['hatch.linewidth'] = 1.0
    # draw_direct()
    # draw_redd()
    draw_transfer()

    # draw_bar()
    # plot_losses()
