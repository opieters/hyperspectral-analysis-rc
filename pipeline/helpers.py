import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime


def check_correlation(df, x, th=0.95, skip=150):
    """
    Checks the pearson correlation with all Series in a DataFrame.

    :param df: DataFrame to compare with
    :param x: Series to compare with
    :param th: Correlation threshold
    :param skip: Amount of samples at the start and end that are skipped in the calculation
    :return: True if correlation below threshold
    """
    for k in df:
        y = df[k].values
        corr, _ = np.abs(pearsonr(x[skip:-skip], y[skip:-skip]))
        if corr > th:
            return False

    return True


def get_vi_nf(df, c_offset=2, verbose=False):
    """
    Compute vegetation indices of the form (x-y)/(x+y)

    The vegetation indices are labeled using the same format as the camera data. This is needed to make the code simpler
    additionally, one can view the VI as a new type of channel, though an artificial one.

    See also :func:`get_x_data()`.

    :param df: DataFrame with source data.
    :param c_offset: camera labeling offset, this is used to make sure there is no overlap with existing data.
    :param verbose: print progress flag
    :return: New DataFrame with VIs
    """
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
                if verbose:
                    print("Included ({}) {}, {}".format(c_offset, a, b))
    return df2


def get_vi_f(df, c_offset=2, verbose=False):
    """
    Compute vegetation indices of the form x/y

    The vegetation indices are labeled using the same format as the camera data. This is needed to make the code simpler
    additionally, one can view the VI as a new type of channel, though an artificial one.

    See also :func:`get_x_data()`.

    :param df: DataFrame with source data.
    :param c_offset: camera labeling offset, this is used to make sure there is no overlap with existing data.
    :param verbose: print progress flag
    :return: New DataFrame with VIs
    """
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
                if verbose:
                    print("Included ({}) {}, {}".format(c_offset, a, b))
    return df2


def mse(x, y):
    """
    Compute the mean square error.

    :param x vector of estimation
    :param y vector or single value of the reference value
    :return: MSE value
    """
    return 1 / len(x) * np.sum(np.power(x - y, 2.0))


def get_performance(X, y, results, x_scaler, model, type):
    """
    Compute the NMSE and MSE of data

    :param X: x data, used to estimate variable
    :param y: target data, used for comparison
    :param results: dictionary to store
    :param x_scaler: StandardScaler object to scale x data
    :param model: Model
    :param type: performance type, used to make keys like "NMSE_{type}".
    :return: result dictionary with NMSE and MSE added
    """
    X = x_scaler.transform(X, copy=True)
    yh = model.predict(X)
    results["NMSE_{}".format(type)] = mse(yh, y) / mse(y, np.mean(y))
    results["MSE_{}".format(type)] = mse(yh, y)

    return results


def get_x_data(df):
    """
    Extract hyperspectral camera data from the DataFrame. This is essentially all columns in the DataFrame that start
    with `c-`.

    :param df: DataFrame with data
    :return: Matrix of x data.
    """
    keys = [i for i in df if i.startswith("c-")]
    keys.sort()
    X, groups = [], []
    for k in keys:
        v = k.split("-")  # format: c-1-c-6-p-9
        # the pixel index v[5] is not used, since it is not always available
        n_camera = int(v[1])
        X.append(df[k].values)

        if len(v) == 5:
            groups.append((n_camera, int(v[3]), int(v[5])))
        else:
            groups.append((n_camera, int(v[3])))
    X = np.stack(X, axis=1)
    return X, groups


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

        te0 = np.arange(t0, t1, np.timedelta64(3, "s"))
        te1 = np.arange(t2, t3, np.timedelta64(3, "s"))
        mask0, mask1, mask2 = t < t0, np.logical_and(t >= t1, t < t2), t >= t3

        dummy_data0 = np.zeros(te0.shape) + 10
        dummy_data1 = np.zeros(te1.shape) + 10
        data_mask = np.concatenate((data_mask[mask0], dummy_data0, data_mask[mask1], dummy_data1, data_mask[mask2]))
        t = np.concatenate((t[mask0], te0, t[mask1], te1, t[mask2]))
        retain_mask = data_mask != 10
    else:
        raise ValueError("Experiment index not supported.")

    dt = t[1:] - t[:-1]
    if np.any(dt != np.timedelta64(3, "s")):
        raise ValueError("There are incorrect time offsets.")

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
