import pandas as pd
import numpy as np

class Parser():
    def __init__(self):
        pass

    def parse(self, exp_dict_list):
        # read exp_dict, read datasets and generate exp_data
        exp_data_list = []
        for exp_dict in exp_dict_list:
            exp_data_list.append(
                self.generate_exp_data(exp_dict)
            )
        return exp_data_list

    def generate_exp_data(self, exp_dict):
        # fill out some exp_dict data
        exp_data = {"name": exp_dict["exp_name"], "type": exp_dict["type"], "datasets": []}

        # number and value of the experimental data points
        ndata = 0
        exp_value = []
        invcov_matrix = np.array([])
        
        # read csv files of different datasets in the experiment and generate input dictionaries 
        for dataset_name in exp_dict["dataset_names"]:
            dataset_dict = self.get_dataset_dict(dataset_name)
            # add number and value of data points to the total number
            ndata += dataset_dict["ndata"]
            exp_value += dataset_dict["value"]
            invcov_matrix = np.append(invcov_matrix, dataset_dict["invcov_matrix"])
            exp_data["datasets"].append(dataset_dict)

        exp_data["ndata"] = ndata
        exp_data["exp_value"] = exp_value

        # read colaboration name, type of the target and projectile from the first dataset
        exp_data["col_name"] = exp_data["datasets"][0]["col_name"]
        exp_data["target"] = exp_data["datasets"][0]["target"]
        exp_data["hadron"] = exp_data["datasets"][0]["hadron"]

        # setup inverse covariance matrix
        if exp_dict["invcov_matrix"] == None:
            exp_data["invcov_matrix"] = np.diag(invcov_matrix)
        else:
            exp_data["invcov_matrix"] = exp_dict["invcov_matrix"]
        
        return exp_data

    def get_dataset_dict(self, name):
        file_name = "./data/datasets/" + name
        # read the csv fiel and translate it to a dictionary
        dataset = pd.read_csv(file_name).to_dict(orient="list")
    
        # generate dataset dictionary
        key_list = ["x", "y", "z", "Q2", "pT"]
        err_list = ["stat_u", "systrel", "systabs_u"]
        dataset_data = dict([(keyname, dataset[keyname]) for keyname in key_list])
        dataset_err = dict([(keyname, dataset[keyname]) for keyname in err_list])
        dataset_value = dataset["value"]

        sigma2 = None
        for i, key in enumerate(dataset_err.keys()):
            if i == 0: sigma2 = np.array(dataset_err[key])**2
            else: sigma2 += np.array(dataset_err[key])**2
        invcov_matrix = 1/sigma2
    
        dataset_dict = {"name": name[0:-4], "col_name": dataset["col"][0], "obs": dataset["obs"][0], "target": dataset["target"][0], 
            "hadron": dataset["hadron"][0], "dependence": dataset["Dependence"][0], "Ebeam": dataset["Ebeam"][0],
            "data": dataset_data, "err": dataset_err, "invcov_matrix": invcov_matrix, "value": dataset_value, "ndata": len(dataset_value)}

        return dataset_dict