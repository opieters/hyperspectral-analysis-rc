#!/usr/bin/env python3

import pickle
import os
import numpy as np
from pipeline.processing import train_augmented_ridge_models
import yaml


def generate_summary(stypes, targets, offsets, src_dir_0, src_dir_1, sum_dir, verbose=False, prefix=""):
    """fn
    Generate summary of model trainings. These aggregate NMSE and MSE values to alleviate the need to load all models
    into memory to find the best one.
    
    :param stypes: split types, e.g. ["plant", "bg1, "bg2"]
    :param targets: target variables, e.g. ["xRH", "Tleaf"]
    :param offsets: data offset range e.g. np.arange(-100, 100, dtype=np.int)
    :param src_dir_0: directory of models for experiment 0
    :param src_dir_1: directory of models for experiment 1
    :param sum_dir: directory for storage of summary reports
    :param verbose: debug print level
    :return: None
    """
    if not os.path.isdir(sum_dir):
        os.makedirs(sum_dir)

    reg_param_trace = np.power(10.0, np.arange(-10, 10))

    model_type = "ridge"
    n_cv = 2

    for target in targets:
        fn_summary = os.path.join(sum_dir, "{}x-summary-{}.pkl".format(prefix, target))

        if os.path.isfile(fn_summary):
            if verbose:
                print("{} found, skipping...".format(fn_summary))
            pass
            continue

        data = dict()

        n_found = 0
        n_missing = 0
        error = False

        for exp, src_dir in enumerate([src_dir_0, src_dir_1]):
            for data_type in stypes:
                for offset in offsets:
                    for augment_data in [True, False]:
                        for reg_param in reg_param_trace:
                            if error:
                                continue

                            fn_result = "{}_{}_{}_{:.2e}_{}.pkl".format(model_type, data_type, target, reg_param, offset)
                            if augment_data:
                                fn_result = "aug_" + fn_result
                            else:
                                fn_result = "naug_" + fn_result

                            fn_result = "{}{}".format(prefix, fn_result)

                            fn_result = os.path.join(src_dir, "results", fn_result)

                            if os.path.isfile(fn_result):
                                if verbose:
                                    print("{} found.".format(fn_result))
                                try:
                                    result = pickle.load(open(fn_result, "rb"))
                                except pickle.UnpicklingError:
                                    print(fn_result, "pickle error.")
                                    error = True
                                    continue
                                key = (exp, data_type, offset, reg_param, augment_data)
                                data[key] = dict()

                                n_found += 1
                                for i in ["train", "val"]:
                                    data[key]["NMSE_{}".format(i)] = \
                                        np.mean([result["NMSE_{}_{}".format(i, j)] for j in range(n_cv)])
                                    data[key]["MSE_{}".format(i)] = \
                                        np.mean([result["MSE_{}_{}".format(i, j)] for j in range(n_cv)])
                                for i in ["train", "test"]:
                                    data[key]["NMSE_final_{}".format(i)] = result["NMSE_{}".format(i)]
                                    data[key]["MSE_final_{}".format(i)] = result["MSE_{}".format(i)]
                            else:
                                n_missing += 1
                                if verbose and ((n_missing + n_found) > 0):
                                    print("{} not found.".format(fn_result))

        if (n_found > 0) and (not error):
            print("{:.2f}% of data found for {}".format(n_found / (n_found + n_missing) * 100, fn_summary))
            pickle.dump(data, open(fn_summary, "wb"))
        else:
            print("No data found for {}.".format(fn_summary))


if __name__ == "__main__":
    config = yaml.load(open("directories.yaml"), Loader=yaml.SafeLoader)

    # possible values for split type
    stypes = ["plant", "bg1", "bg2"]

    # possible target values
    targets = ['xTair', 'xRH', 'Photo', 'Cond', 'Trmmol', 'VpdL', 'Tleaf', 'PARo', 'Press']

    # select offset range
    offsets = np.arange(-1200, 1300, 100, dtype=np.int)
    offsets = [0]

    for augment_data in [True, False]:
        for stype in stypes:
            for target in targets:
                train_augmented_ridge_models(data_type=stype, target=target, offsets=offsets,
                                             augment_data=augment_data, config=config)

    generate_summary(stypes, targets, offsets=offsets, src_dir_0=config["src_dir_0"],
                     src_dir_1=config["src_dir_1"], sum_dir=config["sum_dir"])

