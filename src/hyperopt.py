import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from grid import KGRID2
from config import KMAX, KPTS_H, FLAV

from MCreplicas import MCReplicas
from config import exp_dict_list
from parser import Parser


from data_kfold import DataKFold

"""
Here we perform hyperoptimization of the parameters of the fit. We split the data into n_kfolds, fit the model on (n_kfold - 1) folds 
and use the remaining fold (validation fold) to estimate the quality of the fit, which is characterized by the corrsponding chi2_validation.
We run over all n_kfold possible choices of the validation fold and calculate the final chi2_validation as an average over all
possible choices. The best parameters are estimated as parameters yielding the best final chi2_validation.
The fit is done with just 1 replica, i.e. here we don't generate pseudo data.
"""

if __name__ == "__main__":
    # use Parser to read datasets
    parser = Parser()
    exp_data = parser.parse(exp_dict_list)

    params = {
    "optimizer_name": "Adam",
    "learning_rate": 0.01,
    "initializer": "glorot_normal",
    "max_epochs": 1000,
    "stopping_patience": 0.1,
    "replicas": 1,
    "nodes": [20, FLAV], 
    "activations": ['tanh', 'linear'],
    "nn_depth" : 1,
    "nn_type": "DENSE_FACT",
    "positivity": True,
    "pos_type": "BOTH",
    "n_kfolds": 1,
    "validation_size": 0.25,
    "lambda_undate_freq": 100,
    }

    # create a kfold object
    kfolds = DataKFold(params["n_kfolds"], params["validation_size"])
    # generate masks for kfolds
    training_masks, validation_masks, kfold_masks = kfolds.get_masks(exp_data)

    # generate parameter sets
    params_list = []
    max_epochs_list = [5000]
    lr_list = [0.05]
    nn_depth_list = [2]
    opt_list = ["RMSprop", "Adam", "Adagrad", "Adadelta", "Adamax", "Nadam", "Amsgrad", "SGD"]
    initializer_list = ["glorot_uniform", "glorot_normal"]
    for max_epochs in max_epochs_list:
        for opt in opt_list:
            for lr in lr_list:
                for nn_dpth in nn_depth_list:
                    for init in initializer_list:
                        params_object = deepcopy(params)
                        params_object["max_epochs"] = max_epochs
                        params_object["optimizer_name"] = opt
                        params_object["learning_rate"] = lr
                        params_object["nn_depth"] = nn_dpth
                        params_object["initializer"] = init
                        params_list += [params_object]


    min_chi2pp_idx = 0
    chi2pp_list = []
    for prm_num, prm in enumerate(params_list):
        print("prm_num: ", prm_num)
        chi2pp_kfold = []
        # kfold_num is the number of the validation fold
        for kfold_num in range(params["n_kfolds"]):
            for exp_num, exp_dict in enumerate(exp_data):
                exp_dict["training_mask"] = training_masks[kfold_num][exp_num]
                exp_dict["validation_mask"] = validation_masks[kfold_num][exp_num]
                exp_dict["kfold_mask"] = kfold_masks[kfold_num][exp_num]

            # create the fitter, load exp_data and parameters of the fit, then run the fit
            mcreplicas = MCReplicas(exp_data, prm)
            mcreplicas.run()
            chi2pp_kfold += [mcreplicas.get_chi2pp_kfold()]
        # average chi2pp over validation kfold choice
        chi2pp_kfold = sum(chi2pp_kfold) / params["n_kfolds"]
        # save chi2 for this parameters choice
        chi2pp_list += [chi2pp_kfold]
        if chi2pp_list[-1] < chi2pp_list[min_chi2pp_idx]:
            min_chi2pp_idx = len(chi2pp_list) - 1
        print("prn_num: ", prm_num)
        print("chi2pp_min: ", chi2pp_list[min_chi2pp_idx])
        print("max_epochs: ", params_list[min_chi2pp_idx]["max_epochs"])
        print("optimizer_name: ", params_list[min_chi2pp_idx]["optimizer_name"])
        print("learning_rate: ", params_list[min_chi2pp_idx]["learning_rate"])
        print("nn_depth: ", params_list[min_chi2pp_idx]["nn_depth"])
        print("initializer: ", params_list[min_chi2pp_idx]["initializer"])

    
    chi2pp_min = chi2pp_list[0]
    for i , chi2 in enumerate(chi2pp_list):
        if chi2 < chi2pp_min:
            chi2pp_min = chi2
            min_chi2pp_idx = i

    print(chi2pp_min)
    print("max_epochs: ", params_list[min_chi2pp_idx]["max_epochs"])
    print("optimizer_name: ", params_list[min_chi2pp_idx]["optimizer_name"])
    print("learning_rate: ", params_list[min_chi2pp_idx]["learning_rate"])
    print("nn_depth: ", params_list[min_chi2pp_idx]["nn_depth"])
    print("initializer: ", params_list[min_chi2pp_idx]["initializer"])