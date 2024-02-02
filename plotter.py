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
        "microwave": (68400, 68900),
        "dish washer": (83500, 85100),
        "washing machine": (65200, 67700)
    }
}

GRAPH_NAMES = {
    "SGN": "SGN",
    "AttentionCNN": "Attention CNN",
    "DM_GATE2": "Ours"
}

APP_NAMES = {
    "kettle": "Kettle",
    "fridge": "Refrigerator",
    "microwave": "Microwave oven",
    "dish washer": "Dish washer",
    "washing machine": "Washing machine"
}

# CSFONT = {'fontname': 'Times New Roman'}
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 22}

matplotlib.rc('font', **font)


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


def plot_app(app, df, clf_names, score=True):
    rng = RANGES["ukdale"][app]
    plt.figure(figsize=(12, 6))
    plt.title(APP_NAMES[app], fontweight='bold')

    mains = df.loc[:, "mains"].to_numpy()
    truth = df.loc[:, "truth"].to_numpy()

    x_val = np.arange(0, rng[1] - rng[0], 1)

    plt.plot(x_val, mains[rng[0]: rng[1]], linewidth=1, label="Mains")
    plt.fill_between(x_val, 0, truth[rng[0]:rng[1]],
                     label="Ground truth", color="#aaaaaa77")

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
            print(f"{clf}: {acc:.3f}, F1: {f1:.3f}, Precision: {pre:.3f}, Recall: {recall:.3f}, "
                  f"MAE: {mae:.2f}, SAE: {sae:.2f}")

        plt.plot(x_val, pred[rng[0]: rng[1]], linewidth=1, label=GRAPH_NAMES[clf])

    plt.xlabel("Samples")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper right")

    plt.savefig(f"./figures/{app}.png", bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # df_fridge = load_app("fridge", {
    #     "AttentionCNN": "ukdale-fridge-AttentionCNN.csv",
    #     "SGN": "ukdale-fridge-SGN.csv",
    #     "DM_GATE2": "ukdale-kettle+fridge-DM_GATE2.csv",
    # })
    #
    # print(df_fridge.head())
    # plot_app("fridge", df_fridge, ["AttentionCNN", "SGN", "DM_GATE2"])

    # df_dish = load_app("dish washer", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-dish washer-SGN.csv",
    #     "DM_GATE2": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    # })
    #
    # print(df_dish.head())
    # plot_app("dish washer",
    #          df_dish,
    #          ["AttentionCNN", "SGN", "DM_GATE2"],
    #          score=False)

    # df_kettle = load_app("kettle", {
    #     "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
    #     "SGN": "ukdale-kettle-SGN.csv",
    #     "DM_GATE2": "ukdale-kettle+fridge-DM_GATE2.csv",
    # })
    #
    # print(df_kettle.head())
    # plot_app("kettle",
    #          df_kettle,
    #          ["AttentionCNN", "SGN", "DM_GATE2"],
    #          score=False)

    df_microwave = load_app("microwave", {
        "AttentionCNN": "ukdale-kettle+dish washer+washing machine+microwave-AttentionCNN.csv",
        "SGN": "ukdale-microwave-SGN.csv",
        "DM_GATE2": "ukdale-dish washer+washing machine+microwave-DM_GATE2.csv",
    })

    print(df_microwave.head())
    plot_app("microwave",
             df_microwave,
             ["AttentionCNN", "SGN", "DM_GATE2"],
             score=False)
