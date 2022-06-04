import numpy as np
from observables import AUTsivers, POSObservable
from exp_layer import ExperimentalLayer

import tensorflow as tf
from tensorflow.keras import backend as K

from config import POS_DENSITY, KPTS, KPTS2
from grid import KGRID, POS_INPUT_GRID, POS_SIZE

# list of pdfs we need to calculate observables
pdf_in_obs = {
    "AUTsivers": ["upol_dist", "upol_frag", "sivers_dist"]
}

def get_exp_info(exp_dict):
    """
        exp_info: dict
            contains information about pdf inputs and experimantal layers which construct observables based of pdf outputs
        pdf_input: dict
            dictionary with lists of pdf inputs, where keys are pdf types and elements are list of inputs 
            correponding to different datasets. The expected shape of the input for each dataset is (1, ndata*KPTS2, 2)
        exp_layer: dict
            the elements of this dictionary are observable layers which take pdf outputs and calculate 
            the observables, the keys are names of the experiments. The keys are used to connect pdf outputs
            with exp_layers. The expected size of the pdf output is (1, total_input_size, FLAV)
        total_input_size: dict
            total number of points in pdfs' input/output corresponding to the experement. This number is used to split the total
            pdf output
        exp_name: dict
            contains pdf type and the name of the experiment which we use as a flag to relate pdf output to a particular 
            experimant after splitting
        name: str:
            name of the experiment
        dataset_target: list
            list of data point values corresponding to different dataset. Each element is a TF tensor. It corresponds to one dataset
            and has a shape (1, ndata)
        exp_layer_info: dict
            data used to generate experemental layers
        input_size: dict
            keys are pdf names, elements are lists with the number of points is pdfs' input/output corresponding to each dataset.
            We use it to split total pdfs' output which is feeded to the experemental layer
        dataset_name: dict
            contains names of the dataset. We use it as flags to relate elements of the splitted pdfs' output to datasets
        obs: dict
            the keys are dataset names. They are used as flags to relate splitted pdf outputs to observable classes which are 
            elements of this  dictionary. Each observable class calculates observable predictions ffor this dataset using
            corresponding pdf output
        target: TF tensor
            expected observables values with the Gaussian noice. TF tensor of the shape (1, total number of datapoints in the experement)
    """
    # use this dictionary for the experimental layer
    exp_layer_info = {
        "input_size": {},
        "dataset_name": {},
        "obs": {},
        "target": None,
        "invcov_matrix": None,
        "name": exp_dict["name"],
        "training_mask": None,
        "validation_mask": None,
        "kfold_mask": None,
    }
    
    # construct and return this experimental information
    exp_info = {
        "pdf_input": {},
        "total_input_size": {},
        "exp_name": {},
        "exp_layer": None,
        "dataset_target": {},
        "name": exp_dict["name"],
    }

    # total size of the input provided to pdfs
    exp_total_input_size = 0
    # create a list of targets for all datasets and concatenate it later
    exp_layer_info["target"] = []

    for dataset in exp_dict["datasets"]:
        # get data for each dataset in the experiment
        dataset_info = get_info_dataset(dataset)
        # total size of the pdf inputs
        exp_total_input_size += dataset_info["input_size"]

        for pdf_name in dataset_info["pdf_type"]:
            #initialize some variables of the dictionary
            if pdf_name not in exp_info["pdf_input"].keys():
                # we will use exp_name as a flag to find pdf outputs corresponding to this experiment
                exp_info["exp_name"][pdf_name] = [exp_dict["name"]]
                # pdf_input, input_size and dataset name corresponding to each dataset
                exp_info["pdf_input"][pdf_name] = []
                exp_layer_info["input_size"][pdf_name] = []
                exp_layer_info["dataset_name"][pdf_name] = []

            # list of dataset pdf_inputs
            exp_info["pdf_input"][pdf_name] += [dataset_info["pdf_input"][pdf_name]]
            # we will use input size and name of the dataset to find pdf outputs for different experimental layers
            exp_layer_info["input_size"][pdf_name] += [dataset_info["input_size"]]
            exp_layer_info["dataset_name"][pdf_name] += [dataset_info["dataset_name"]]

        # add dataset observable and use it later in the experimental layer
        exp_layer_info["obs"].update({dataset_info["dataset_name"]: dataset_info["obs"]})
        # add values of the data points
        exp_layer_info["target"] += [dataset_info["target"]]
        exp_info["dataset_target"][dataset_info["dataset_name"]] = dataset_info["target"]

    # expected shape of the exp_layer_info["target"] is (1, exp_total_input_size)
    exp_layer_info["target"] = K.concatenate(exp_layer_info["target"], axis=-1)

    # inverse covariance matrix of the experiment and the number of data point
    exp_layer_info["invcov_matrix"] = K.constant(exp_dict["invcov_matrix"])
    for mask_key in ["training_mask", "validation_mask", "kfold_mask"]:
        exp_layer_info[mask_key] = K.constant(exp_dict[mask_key])
        exp_layer_info[mask_key] = tf.expand_dims(exp_layer_info[mask_key], axis=0)
    exp_layer_info["ndata"] = exp_dict["ndata"]

    # we will use this to separate pdf outputs between different experimental layers
    for key in exp_info["pdf_input"]:
        exp_info["total_input_size"][key] = [exp_total_input_size]
    # generate the experimental layer
    exp_info["exp_layer"] = {exp_dict["name"]: ExperimentalLayer(exp_layer_info)}

    return exp_info

def get_info_dataset(dataset):
    """
        dataset_info: dict
            contains all relevant information about dataset
        pdf_input: dict
            dictionary with the pdf input, where keys are pdf types and elements are inputs 
            correponding to the dataset. The expected shape of the input is (1, ndata*KPTS, 2)
        pdf_type: list
            list of pdf types that we need to calculate the observables for this dataset
        input_size: int
            number of points in the pdf input, it is ndata*KPTS2 except the positivity dataset where it is pos_density*(KPTS//2)
        dataset_name: str
            name of the dataset
        obs: class
            observables class that we use to calculate observables for this dataset
        target:
            TF tesor with expected values of the observables with the Gaussina noice, shape=(1,ndata)
    """
    dataset_info = {
        "pdf_input": {},
        "pdf_type": [],
        "input_size": 0,
        "dataset_name": dataset["name"],
        "obs": None,
        "target": None,
    }

    # datasets of different types are processed differently
    if dataset["obs"] == "positivity":
        dataset_info["input_size"] = POS_SIZE
        dataset_info["pdf_type"] = dataset["pdf_type"]

        # add positivity input grid for each pdf
        for pdf_name in dataset_info["pdf_type"]:
            dataset_info["pdf_input"][pdf_name] = POS_INPUT_GRID

        dataset_info["obs"] = POSObservable(dataset_info["pdf_type"], dataset["ndata"], dataset["pos_type"])

        dataset_info["target"] = K.constant(np.zeros(shape=(1,POS_DENSITY)))

    else:
        target = dataset["target"]
        hadron = dataset["hadron"]

        dataset_info["input_size"] = dataset["ndata"]*KPTS2
        input_grid_dist = np.zeros(shape=(1,dataset_info["input_size"],2))
        input_grid_frag = np.zeros(shape=(1,dataset_info["input_size"],2))

        # construct input grades for pdfs and fragmentation functions
        for n, val in enumerate(zip(dataset["data"]["x"], dataset["data"]["z"], dataset["data"]["pT"])):
            xB, zh, pTx, pTy = val[0], val[1], val[2], 0
            for i, kx in enumerate(KGRID):
                for j, ky in enumerate(KGRID):
                    index = n*KPTS2 + i*KPTS + j
                    input_grid_dist[0][index][0] = xB
                    input_grid_dist[0][index][1] = kx**2 + ky**2
                    input_grid_frag[0][index][0] = zh
                    input_grid_frag[0][index][1] = (pTx  - zh*kx)**2 + (pTy  - zh*ky)**2

        for pdf_name in pdf_in_obs[dataset["obs"]]:
            if "dist" in pdf_name:
                dataset_info["pdf_type"].append(pdf_name + "_" + target)
            else:
                dataset_info["pdf_type"].append(pdf_name + "_" + hadron)

        for pdf_name in dataset_info["pdf_type"]:
            if "dist" in pdf_name:
                dataset_info["pdf_input"][pdf_name] = input_grid_dist
            else:
                dataset_info["pdf_input"][pdf_name] = input_grid_frag

        if dataset["obs"]=="AUTsivers":
            dataset_info["obs"] = AUTsivers(dataset)

        # true value of the experimental data points, expected shape=(1, ndata)
        dataset_info["target"] = tf.expand_dims(K.constant(dataset["value"]), axis=0)

    return dataset_info
    
