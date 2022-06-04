import numpy as np
import matplotlib.pyplot as plt
from grid import KGRID2
from config import KMAX, KPTS_H

from MCreplicas import MCReplicas
from parser import Parser
from config import params, exp_dict_list

from data_kfold import DataKFold

if __name__ == "__main__":
    # use Parser to read datasets
    parser = Parser()
    exp_data = parser.parse(exp_dict_list)

    kfolds = DataKFold(params["n_kfolds"], params["validation_size"])
    training_masks, validation_masks, kfold_masks = kfolds.get_masks(exp_data)

    for kfold_num in range(params["n_kfolds"]):
        for exp_num, exp_dict in enumerate(exp_data):
            exp_dict["training_mask"] = training_masks[kfold_num][exp_num]
            exp_dict["validation_mask"] = validation_masks[kfold_num][exp_num]
            exp_dict["kfold_mask"] = kfold_masks[kfold_num][exp_num]

        # create fitter and load exp_data and parameters of the fit, then run the fit
        mcreplicas = MCReplicas(exp_data, params)
        mcreplicas.run()

    # obtain prediction for the dataset "2002"
    list_of_datasets = mcreplicas.get_dataset_list()
    obs_target = mcreplicas.get_target("2002")
    obs_pred_mean = mcreplicas.get_obs_pred_mean("2002")
    obs_pred_sigma = np.sqrt(mcreplicas.get_obs_pred_variance("2002"))

    obs_name = mcreplicas.get_obs_name("2002")
    dep_variable_name = mcreplicas.get_dep_variable_name("2002")
    dep_variable_values = mcreplicas.get_dep_variable_values("2002")

    # plot the result
    plt.figure(figsize=(5,5))
    plt.scatter(dep_variable_values, obs_target, label="data")
    plt.scatter(dep_variable_values, obs_pred_mean, label="prediction", color='C1')
    #plt.errorbar(dep_variable_values, obs_pred_mean, yerr=obs_pred_sigma, fmt="o", color='C1')
    plt.xlabel(dep_variable_name)
    plt.ylabel("AUTSivers")
    plt.legend()
    plt.show()

    # plot "u" and "d" tmds
    transverse_part_u = [mcreplicas.get_transverse_part("upol_dist_proton", "u", k2) for k2 in KGRID2]
    tr_part_u_rep = []
    for rnum in range(mcreplicas.get_nrep()):
        tr_part_u_rep += [[mcreplicas.get_transverse_part_rep(rnum, "upol_dist_proton", "u", k2) for k2 in KGRID2]]
    kgrid = np.linspace(0, KMAX, KPTS_H)

    plt.figure(figsize=(5,5))
    plt.plot(kgrid, transverse_part_u, label="distribution function")
    for rnum in range(mcreplicas.get_nrep()):
        plt.plot(kgrid, tr_part_u_rep[rnum], alpha = 0.2, color='C1')

    plt.xlabel("k2")
    plt.ylabel("f(k2)")

    plt.legend()
    plt.show()

    print("Hello world!")