from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, \
    accuracy_score, precision_score, matthews_corrcoef
import numpy as np
import nilmtk.utils as utils

# on_threhold = {'fridge': 50, 'kettle': 2000, 'dish washer': 50, 'washing machine': 40,
#                'microwave': 200,
#                'drill': 0}
on_threhold = {app: data['on'] for app, data in utils.GENERAL_APP_META.items()}


SAE_DIV_LEN = 1200


def sae(app_name, app_gt, app_pred):
    sae_sum = 0

    n_seg = len(app_gt) // SAE_DIV_LEN
    for i in range(n_seg):
        idx = i * SAE_DIV_LEN
        gt = app_gt[idx:idx + SAE_DIV_LEN]
        pred = app_pred[idx:idx + SAE_DIV_LEN]
        gt_sum = np.sum(gt)
        pred_sum = np.sum(pred)
        diff = np.abs(pred_sum - gt_sum)
        sae_sum += diff / SAE_DIV_LEN

    return sae_sum / n_seg


def mae(app_name, app_gt, app_pred):
    return mean_absolute_error(app_gt, app_pred)


def rmae(app_name, app_gt, app_pred):
    constant = 1
    numerator = np.abs(app_gt - app_pred)
    max_temp = np.where(app_gt > app_pred, app_gt, app_pred)
    denominator = constant + max_temp
    return np.mean(numerator / denominator)


def nep(app_name, app_gt, app_pred):
    numerator = np.sum(np.abs(app_gt - app_pred))
    denominator = np.sum(np.abs(app_gt))
    return numerator / denominator


def rmse(app_name, app_gt, app_pred):
    return mean_squared_error(app_gt, app_pred) ** (.5)


def recall(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)

    return recall_score(gt_temp, pred_temp)


def precision(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)

    return precision_score(gt_temp, pred_temp)


def accuracy(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)

    return accuracy_score(gt_temp, pred_temp)


def f1score(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)

    return f1_score(gt_temp, pred_temp)


def MCC(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)

    return matthews_corrcoef(gt_temp, pred_temp)


def omae(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)
    pred_temp = pred_temp[idx]

    return mae(app_name, gt_temp, pred_temp)


def ormae(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)
    pred_temp = pred_temp[idx]

    return rmae(gt_temp, pred_temp)
