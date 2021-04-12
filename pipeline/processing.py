#!/usr/bin/env python3

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import pickle
import os
import yaml
import numpy as np
import scipy.signal as signal
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime
from urllib import request
from pipeline.helpers import get_vi_f, get_vi_nf


def download_data(fn, data_dir):
    url = "https://zenodo.org/record/3897289/files/{}?download=1".format(fn)
    fn = os.path.join(data_dir, fn)
    if not os.path.isfile(fn):
        print("Downloading data from", url)
        request.urlretrieve(url, fn)


def preprocess_data(df, augment):
    rm_idx = np.concatenate((np.arange(0, 300, dtype=np.int), np.arange(len(df) - 300, len(df))))

    df = df.drop(index=rm_idx)

    if augment:
        df_vi_f = get_vi_f(df, c_offset=2, verbose=False)
        df = pd.concat([df, df_vi_f], axis=1, sort=False)
        df_vi_nf = get_vi_nf(df, c_offset=2 + 41, verbose=False)
        df = pd.concat([df, df_vi_nf], axis=1, sort=False)

    return df


def load_data(src_dir, data_dir, exp_idx, data_type, augment_data):
    cache_dir = os.path.join(src_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    if augment_data:
        fn = "x-df-augmented-{}.pkl".format(data_type)
        fn = os.path.join(src_dir, "cache", fn)
    else:
        fn = "x-df-{}.pkl".format(data_type)
        fn = os.path.join(src_dir, "cache", fn)
    if not os.path.exists(fn):
        if (exp_idx == 0) and (data_type == "bg1"):
            data_type = "pvc"
        elif (exp_idx == 0) and (data_type == "bg2"):
            data_type = "wood"
        elif (exp_idx == 1) and (data_type == "bg1"):
            data_type = "cotton"
        elif (exp_idx == 1) and (data_type == "bg2"):
            data_type = "ytong"
        csv_fn = "exp-{}-{}.csv".format(exp_idx+1, data_type)
        download_data(fn=csv_fn, data_dir=data_dir)
        df = pd.read_csv(os.path.join(data_dir, csv_fn))
        df["HHMMSS"] = pd.to_datetime(df["HHMMSS"])
        df = preprocess_data(df, augment=augment_data)
        pickle.dump(df, open(fn, "wb"))
    else:
        df = pickle.load(open(fn, "rb"))

    return df


def get_x_data(df, use_filter=False, camera_split=None):
    """
    Extract x data for training from DataFrame. This is essentially all columns in the DataFrame that start
    with `c-`.

    :param df: Source DataFrame with all data
    :param use_filter: use low-pass filter on the data flag
    :param camera_split: Extract only data from a specific camera. None uses all camera data, 0 or 1 select NIR and VIS
    cameras respecively.
    :return: X, groups: x-data and group indices from DatFrame
    """
    keys = [i for i in df if i.startswith("c-")]
    keys.sort()
    X, groups = [], []
    for k in keys:
        v = k.split("-")  # format: c-1-c-6-p-9
        n_camera = int(v[1])
        if camera_split is None:
            if use_filter:
                X.append(filter_data(df[k].values))
            else:
                X.append(df[k].values)
        elif n_camera == camera_split:
            if use_filter:
                X.append(filter_data(df[k].values))
            else:
                X.append(df[k].values)

        if len(v) == 5:
            groups.append((n_camera, int(v[3]), int(v[5])))
        else:
            groups.append((n_camera, int(v[3])))
    X = np.stack(X, axis=1)
    return X, groups


def mse(x, y):
    """
    Compute the MSE.

    :param x: data trace 1
    :param y: data trace 2 or single value
    :return: MSE value
    """
    return 1 / len(x) * np.sum(np.power(x - y, 2.0))


def get_masks(data_length, batch_size, mask_type, offset, t, exp_idx):
    """
    Get the data masks used for training, validation and testing.

    :param data_length: total length of the data
    :param batch_size: split batch size, two batches are aggregated to form a train/val/test batch.
    :param mask_type: type of mask to generate. Can be either `train` for train data, `val` for validation data, `test`
    for test data, `val+train` for validation and train data, `all` for test, train and validation data and `none` for
    all data. This final key is useful in case there is an offset, because of the missing data that might otherwise
    generate incorrect data
    :param offset: offset between x and y data.
    :param t: time values
    :param exp_idx: experiment index, either `0` or `1`
    :return: x and y masks as a tuple.
    """

    assert data_length == len(t)
    aggregate = 2
    data_mask = np.arange(data_length, dtype=np.int)
    data_mask = data_mask // batch_size
    data_mask = data_mask % (3 + aggregate * 3)

    if offset > (aggregate * batch_size):
        raise ValueError("Offset cannot be greater than the batch size.")

    if exp_idx == 0:
        t0 = np.datetime64(datetime(year=2019, month=3, day=19, hour=7, minute=53, second=13))
        t1 = np.datetime64(datetime(year=2019, month=3, day=19, hour=9, minute=33, second=13))

        mask0, mask1 = t < t0, t >= t1
        te = np.arange(t0, t1, np.timedelta64(3, "s"))
        dummy_data = np.zeros(te.shape) + 10
        data_mask = np.concatenate((data_mask[mask0], dummy_data, data_mask[mask1]))
        t = np.concatenate((t[mask0], te, t[mask1]))
        retain_mask = data_mask != 10
    elif exp_idx == 1:
        t0 = np.datetime64(datetime(year=2019, month=3, day=30, hour=9, minute=27, second=47))
        t1 = np.datetime64(datetime(year=2019, month=3, day=30, hour=12, minute=47, second=47))
        t2 = np.datetime64(datetime(year=2019, month=3, day=30, hour=17, minute=22, second=47))
        t3 = np.datetime64(datetime(year=2019, month=3, day=30, hour=19, minute=2, second=47))
        t4 = np.datetime64(datetime(year=2019, month=3, day=31, hour=3, minute=56, second=20))
        t5 = np.datetime64(datetime(year=2019, month=3, day=31, hour=3, minute=59, second=50))

        te0 = np.arange(t0, t1, np.timedelta64(3, "s"))
        te1 = np.arange(t2, t3, np.timedelta64(3, "s"))
        te2 = np.arange(t4, t5, np.timedelta64(3, "s"))
        mask0, mask1, mask2, mask3 = t < t0, np.logical_and(t >= t1, t < t2), np.logical_and(t >= t3, t < t4), t >= t5

        dummy_data0 = np.zeros(te0.shape) + 10
        dummy_data1 = np.zeros(te1.shape) + 10
        dummy_data2 = np.zeros(te2.shape) + 11
        data_mask = np.concatenate((data_mask[mask0], dummy_data0, data_mask[mask1], dummy_data1, data_mask[mask2],
                                    dummy_data2, data_mask[mask3]))
        t = np.concatenate((t[mask0], te0, t[mask1], te1, t[mask2], te2, t[mask3]))
        retain_mask = data_mask != 10
    else:
        raise ValueError("Experiment index not supported.")

    dt = t[1:] - t[:-1]
    #if np.any(dt != np.timedelta64(3, "s")):
    #    raise ValueError("There are incorrect time offsets.")

    if mask_type == "test":
        idx = [0, 1]
    elif mask_type == "val":
        idx = [3, 4]
    elif mask_type == "train":
        idx = [6, 7]
    elif mask_type == "train+val":
        idx = [3, 4, 6, 7]
    elif mask_type == "all":
        idx = [0,1,3,4,6,7]
    elif mask_type == "none":
        idx = [0,1,2,3,4,5,6,7, 8]
    else:
        raise ValueError("Unknown mask type {}".format(mask_type))
    for i in idx:
        data_mask[data_mask == i] = idx[0]

    mask_x, mask_y = np.copy(data_mask), np.copy(data_mask)
    mask_y = np.roll(mask_y, offset, axis=0)

    mask_x = np.logical_and(mask_x == idx[0], mask_y == idx[0])
    mask_y = np.roll(mask_x, -offset)

    mask_x = mask_x[retain_mask]
    mask_y = mask_y[retain_mask]

    return mask_x, mask_y


def filter_data(x):
    """
    Low-pass filter data using heuristically chosen filter response that should only reduce the noise.

    :param x: data trace
    :return: filtered data trace
    """
    filter = {
        "fs": 1 / 3,
        "numtaps": 50,
        "bands": [0, 0.01, 0.02, 1 / 3 / 2],
        "desired": [1, 0],
    }

    coeffs = signal.remez(**filter)

    # set incorrect values to known values
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0

    x = np.convolve(x, coeffs, "same")
    return x


def get_performance(X, y, results, x_scaler, model, type, verbose=False):
    X = x_scaler.transform(X)
    if verbose:
        print("Predicting data... ", end="")
    yh = model.predict(X)
    results["NMSE_{}".format(type)] = mse(yh, y) / mse(y, np.mean(y))
    results["MSE_{}".format(type)] = mse(yh, y)
    if verbose:
        print("Done.")

    return results


def cv_train(x_train, y_train, x_val, y_val, model_type, model_params, results):
    """

    :param x_train: x train data
    :param y_train: y train data
    :param x_val: x validation data
    :param y_val: y validation data
    :param model_type: model type (ridge or lasso)
    :param model_params: model parameters
    :param results: dictionary to store results
    :return: result dictionary with added new results
    """

    x_train, y_train, x_val, y_val = np.copy(x_train), np.copy(y_train), np.copy(x_val), np.copy(y_val)
    for i, xt, yt, xv, yv in zip(range(2), [x_train, x_val], [y_train, y_val], [x_val, x_train], [y_val, y_train]):
        x_scaler = StandardScaler()
        xt_tf = x_scaler.fit_transform(xt)
        results["x_scaler_{}".format(i)] = x_scaler

        if model_type == "lasso":
            model = Lasso(**model_params)
        elif model_type == "ridge":
            model = Ridge(**model_params)
        else:
            raise ValueError("Unknown model {}.".format(model_type))
        model.fit(xt_tf, yt)
        results["model_{}".format(i)] = model

        # compute train data performance
        results = get_performance(xt, yt, results, x_scaler, model, "train_{}".format(i))

        # compute validation data performance
        results = get_performance(xv, yv, results, x_scaler, model, "val_{}".format(i))

    return results


def final_train(x_train, y_train, x_test, y_test, model_type, model_params, results):
    x_scaler = StandardScaler()
    x_train_tf = x_scaler.fit_transform(x_train)
    results["x_scaler"] = x_scaler

    if model_type == "lasso":
        model = Lasso(**model_params)
    elif model_type == "ridge":
        model = Ridge(**model_params)
    else:
        raise ValueError("Unknown model {}.".format(model_type))
    model.fit(x_train_tf, y_train)
    results["model"] = model

    # compute train data performance
    results = get_performance(x_train, y_train, results, x_scaler, model, "train")

    # compute test data performance
    results = get_performance(x_test, y_test, results, x_scaler, model, "test")

    return results


def check_correlation(df, x, th=0.95, skip=150):
    for k in df:
        y = df[k].values
        corr, _ = np.abs(pearsonr(x[skip:-skip], y[skip:-skip]))
        if corr > th:
            return False

    return True


def data_augmentation_a(df, c_offset=2):
    df2 = pd.DataFrame()
    for a in range(41):
        for b in range(a + 1, 41):
            c_a = a // 25
            c_b = b // 25

            ch_a = a - 25 * c_a
            ch_b = b - 25 * c_b

            x = df["c-{}-c-{}".format(c_a, ch_a)]
            y = df["c-{}-c-{}".format(c_b, ch_b)]

            z = (x - y) / (x + y)

            if check_correlation(df2, z, th=.95):
                df2["c-{}-c-{}".format(a + c_offset, b)] = z
                print("Included ({}) {}, {}".format(c_offset, a, b))
    return df2


def data_augmentation_b(df, c_offset=2):
    df2 = pd.DataFrame()
    for a in range(41):
        for b in range(41):
            if a == b:
                continue
            c_a = a // 25
            c_b = b // 25

            ch_a = a - 25 * c_a
            ch_b = b - 25 * c_b

            x = df["c-{}-c-{}".format(c_a, ch_a)]
            y = df["c-{}-c-{}".format(c_b, ch_b)]

            z = x / y

            if check_correlation(df2, z, th=.95):
                df2["c-{}-c-{}".format(a + c_offset, b)] = z
                print("Included ({}) {}, {}".format(c_offset, a, b))
    return df2


def run_pipeline(df, target, model_type, model_params, offset, batch_size, camera_split, exp_idx):
    initial_offset = 300

    X, groups = get_x_data(df, use_filter=False, camera_split=camera_split)
    y = df[target].values
    t = df["HHMMSS"].values

    # remove first samples since there might be an issue here with missing data depending on offsets in the
    # system
    X, y, t = X[initial_offset:-initial_offset], y[initial_offset:-initial_offset], t[initial_offset:-initial_offset]

    data_length = len(X)

    mask_x, mask_y = get_masks(data_length=data_length, batch_size=batch_size, mask_type="train", offset=offset,
                               exp_idx=exp_idx, t=t)
    x_train, y_train = X[mask_x], y[mask_y]

    mask_x, mask_y = get_masks(data_length=data_length, batch_size=batch_size, mask_type="val", offset=offset,
                               exp_idx=exp_idx, t=t)
    x_val, y_val = X[mask_x], y[mask_y]

    mask_x, mask_y = get_masks(data_length=data_length, batch_size=batch_size, mask_type="test", offset=offset,
                               exp_idx=exp_idx, t=t)
    x_test, y_test = X[mask_x], y[mask_y]
    # compute
    results = dict()

    results = cv_train(x_train, y_train, x_val, y_val, model_type, model_params=model_params, results=results)

    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    results = final_train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_type=model_type,
                          model_params=model_params, results=results)

    return results


def augment_dataframe(df):
    rm_keys = ['FTime', 'EBal?', 'Area', 'BLC_1', 'StmRat', 'BLCond', 'TBlk', 'Flow', 'CsMch', 'HsMch',
               'StableF', 'BLCslope', 'BLCoffst', 'f_parin', 'f_parout', 'alphaK', 'Status']

    df = df.drop(columns=rm_keys)

    df2 = data_augmentation_a(df, c_offset=2)
    df = pd.concat([df, df2], axis=1, sort=False)

    df3 = data_augmentation_b(df, c_offset=41 + 2)
    df = pd.concat([df, df3], axis=1, sort=False)

    return df


def train_augmented_lasso_models(data_type, target, offsets, augment_data=True):
    reg_param_trace = np.power(10.0, np.arange(-5, 5, 0.5))
    __train_augmented_models__(model_type="lasso", data_type=data_type, target=target,
                               offsets=offsets, augment_data=augment_data,
                               reg_param_trace=reg_param_trace)


def train_augmented_ridge_models(data_type, target, offsets, config, augment_data=True):
    reg_param_trace = np.power(10.0, np.arange(-8, 9))
    __train_augmented_models__(model_type="ridge", data_type=data_type, target=target,
                               offsets=offsets, augment_data=augment_data,
                               reg_param_trace=reg_param_trace, config=config)


def __train_augmented_models__(model_type, data_type, target, offsets, augment_data, reg_param_trace, config):

    result_dir = "results/"

    src_dirs = [config["src_dir_0"], config["src_dir_1"]]
    dst_dirs = [config["dst_dir_0"], config["dst_dir_1"]]

    batch_size = 1500

    for exp_idx in range(len(src_dirs)):
        src_dir, dst_dir = src_dirs[exp_idx], dst_dirs[exp_idx]

        if not os.path.isdir(os.path.join(dst_dir, result_dir)):
            os.makedirs(os.path.join(dst_dir, result_dir))

        df = load_data(src_dir=src_dir, data_dir=config["data_dir"], exp_idx=exp_idx, data_type=data_type, augment_data=augment_data)

        for offset in offsets:
            print("Offset:", offset)
            for reg_param in reg_param_trace:
                print("Reg param: {:.2e}...".format(reg_param))

                if augment_data:
                    fn_result = "aug_{}_{}_{}_{:.2e}_{}.pkl".format(model_type, data_type, target, reg_param, offset)
                else:
                    fn_result = "naug_{}_{}_{}_{:.2e}_{}.pkl".format(model_type, data_type, target, reg_param, offset)

                fn_result = os.path.join(dst_dir, result_dir, fn_result)

                if os.path.isfile(fn_result):
                    print("Result file already exists...")
                    print()
                    continue

                model_params = {
                    "alpha": reg_param,
                    "normalize": False,
                    "random_state": 25,
                    # "selection": "random",
                    # "max_iter": 10000
                }

                results = run_pipeline(df=df, target=target, model_type=model_type, model_params=model_params,
                                       offset=offset, batch_size=batch_size, camera_split=None, exp_idx=exp_idx)

                # save results to folder
                pickle.dump(results, open(fn_result, "wb"))

                print("Saved results.")
                print()

        exp_idx += 1

    print("Done.")


def train_ridge_models(data_size, data_type, target, index, camera_split, offsets):
    model_type = "ridge"

    config = yaml.load(open("../directories.yaml"), Loader=yaml.BaseLoader)

    result_dir = "results/"

    src_dir_0 = config["src_dir_0"]
    src_dir_1 = config["src_dir_1"]
    dst_dir_0 = config["dst_dir_0"]
    dst_dir_1 = config["dst_dir_1"]

    reg_param_trace = np.power(10.0, np.arange(-10, 11))

    batch_size = 1500  # 3*150 minutes

    exp_idx = 0

    for src_dir, dst_dir in zip([src_dir_0, src_dir_1], [dst_dir_0, dst_dir_1]):
        print("Experiment", exp_idx)
        df_filename = os.path.join(src_dir, "cache",
                                   "{}-{}-df-{}-{}-{}.pkl".format(data_type, 1, *data_size, index))

        df = pickle.load(open(df_filename, "rb"))

        for offset in offsets:
            print("Offset:", offset)
            for reg_param in reg_param_trace:
                print("Reg param: {:.2e}...".format(reg_param))

                model_params = {
                    "alpha": reg_param,
                    "normalize": False,
                    "random_state": 25,
                }

                if camera_split is not None:
                    fn_result = "{}_{}_{}_{}_{}_{}_{:.2e}_{}_{}.pkl".format(model_type, index, data_type,
                                                                            *data_size, target, reg_param,
                                                                            offset, camera_split)
                else:
                    fn_result = "{}_{}_{}_{}_{}_{}_{:.2e}_{}.pkl".format(model_type, index, data_type,
                                                                         *data_size, target, reg_param,
                                                                         offset)

                fn_result = os.path.join(dst_dir, result_dir, fn_result)

                if os.path.isfile(fn_result):
                    print("Result file already exists...")
                    print()
                    continue

                results = run_pipeline(df, target, model_type, model_params, offset, batch_size, camera_split, exp_idx=exp_idx)

                # save results to folder
                pickle.dump(results, open(fn_result, "wb"))

                print("Saved results.")
                print()

        exp_idx += 1

    print("Done.")
