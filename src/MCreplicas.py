from copy import deepcopy
from fitter import Fitter
import numpy as np
from numpy.random import rand
from scipy.stats import multivariate_normal
from config import POS_DENSITY, XPOS_MIN, XPOS_MAX, FLAV, KPTS_H
from grid import KGRID2

from scipy.interpolate import RectBivariateSpline, UnivariateSpline

class MCReplicas():
    def __init__(self, exp_data, params):
        self.exp_data = exp_data
        self.params = params

        self.nrep = self.params["replicas"]
        if self.nrep == 1: self.randomize = False
        else: self.randomize = True

        self.replicas = [self.get_mc(self.randomize) for i in range(self.nrep)]
        self.rep_fitters = [Fitter(self.replicas[i], self.params) for i in range(self.nrep)]
        self.obs_pred = [None for i in range(self.nrep)]
        self.obs_pred_mean = {}
        self.obs_pred_variance = {}
        self.chi2pp_validation = None
        self.chi2pp_kfold = None
        self.target_original = {}

        # some parameters of about datasets
        self.obs_name = {}
        self.dep_variable_name = {}
        self.dep_variable_values = {}

        # list of pdfs and flavors in the fit
        self.pdf_list = None
        self.flav_list = None

        # average pdf over replicas
        self.pdf_pred = {}

        for exp_dict in self.exp_data:
            for dataset in exp_dict["datasets"]:
                self.target_original.update({dataset["name"]:np.expand_dims(dataset["value"], axis=0)})
                self.obs_name.update({dataset["name"]:dataset["obs"]})
                self.dep_variable_name.update({dataset["name"]:dataset["dependence"]})
                self.dep_variable_values.update({dataset["name"]:dataset["data"][dataset["dependence"]]})

    def get_mc(self, randomize=True):
        mc_data = deepcopy(self.exp_data)
        for exp_dict in mc_data:
            # store original experimental point
            exp_dict["exp_value_original"] = exp_dict["exp_value"]
            if exp_dict["name"] != "pos_exp":
                if randomize :
                    # calculate covariance matrix
                    cov_matrix = np.linalg.inv(exp_dict["invcov_matrix"])
                    # sample from the multi-Gaussian distribution
                    exp_dict["exp_value"] = list(multivariate_normal.rvs(mean=exp_dict["exp_value_original"], cov=cov_matrix, size=1))
            datapoint_flag = 0
            # update dataset dictionaries
            for dataset in exp_dict["datasets"]:
                dataset["value_original"] = dataset["value"]
                if randomize:
                    dataset["value"] = exp_dict["exp_value"][datapoint_flag:(dataset["ndata"]+datapoint_flag)]
                datapoint_flag += dataset["ndata"]

        return mc_data

    def run(self):
        # run and obtain predictions for all replicas
        for i in range(self.nrep):
            print(f"Replica {i+1}/{self.nrep}")
            self.rep_fitters[i].run()
            self.obs_pred[i] = self.rep_fitters[i].get_obs_pred()

        # calculate average observables over all replicas
        for key in self.obs_pred[0]:
            self.obs_pred_mean[key] = deepcopy(self.obs_pred[0][key])
        
        for i in range(1, self.nrep):
            for key in self.obs_pred_mean.keys():
                self.obs_pred_mean[key] += self.obs_pred[i][key]

        for key in self.obs_pred_mean.keys():
            self.obs_pred_mean[key] /= self.nrep

        # calculate variance of the obsetvables
        for key in self.obs_pred_mean.keys():
            self.obs_pred_variance[key] = (self.obs_pred[0][key] - self.obs_pred_mean[key])**2

        for i in range(1, self.nrep):
            for key in self.obs_pred_variance.keys():
                self.obs_pred_variance[key] += (self.obs_pred[i][key] - self.obs_pred_mean[key])**2

        for key in self.obs_pred_variance.keys():
            self.obs_pred_variance[key] /= self.nrep

        # calculate chi2pp_validation and chi2pp_kfold
        self.chi2pp_validation = 0
        self.chi2pp_kfold = 0
        for i in range(self.nrep):
            self.chi2pp_validation += self.rep_fitters[i].get_chi2pp_validation()
            self.chi2pp_kfold += self.rep_fitters[i].get_chi2pp_kfold()
        self.chi2pp_validation /= self.nrep
        self.chi2pp_kfold /= self.nrep

        # list of pdfs and flavours
        self.pdf_list = self.rep_fitters[0].get_pdf_list()
        self.flav_list = self.rep_fitters[0].get_flav_list()
        
        # average pdfs over all replicas
        for pdf_name in self.pdf_list:
            self.pdf_pred[pdf_name] = {}

            # construct and save full pdf spline
            xspace = np.linspace(XPOS_MIN, XPOS_MAX, POS_DENSITY)
            data = np.zeros(shape=(FLAV, POS_DENSITY, KPTS_H))
            
            # fill out data tensor for spline
            for fnum, flav_name in enumerate(self.flav_list):
                for xnum, x in enumerate(xspace):
                    for knum, k2 in enumerate(KGRID2):
                        for rnum in range(self.nrep):
                            data[fnum][xnum][knum] += self.get_full_pdf_rep(rnum, pdf_name, flav_name, x, k2)

            data /= self.nrep

            self.pdf_pred[pdf_name]["full_pdf"] = {}
            for fnum, flav_name in enumerate(self.flav_list):
                self.pdf_pred[pdf_name]["full_pdf"][flav_name] = RectBivariateSpline(xspace, KGRID2, data[fnum], kx=3, ky=4)

            # construct and save collinear part spline
            data = np.zeros(shape=(FLAV, POS_DENSITY))
            # fill out data tensor for spline
            for fnum, flav_name in enumerate(self.flav_list):
                for xnum, x in enumerate(xspace):
                    for rnum in range(self.nrep):
                        data[fnum][xnum] += self.get_coll_part_rep(rnum, pdf_name, flav_name, x)

            data /= self.nrep

            self.pdf_pred[pdf_name]["coll_part"] = {}
            for fnum, flav_name in enumerate(self.flav_list):
                self.pdf_pred[pdf_name]["coll_part"][flav_name] = UnivariateSpline(xspace, data[fnum], k=5)

            # construct and save k dependence spline
            data = np.zeros(shape=(FLAV, KPTS_H))
            # fill out data tensor for spline
            for fnum, flav_name in enumerate(self.flav_list):
                for knum, k2 in enumerate(KGRID2):
                    for rnum in range(self.nrep):
                        data[fnum][knum] += self.get_transverse_part_rep(rnum, pdf_name, flav_name, k2)

            data /= self.nrep

            self.pdf_pred[pdf_name]["branch_k"] = {}
            for fnum, flav_name in enumerate(self.flav_list):
                self.pdf_pred[pdf_name]["branch_k"][flav_name] = UnivariateSpline(KGRID2, data[fnum], k=5)        

        return self.obs_pred_mean

    def get_dataset_list(self):
        return list(self.target_original.keys())

    def get_target(self, dataset_name):
        return self.target_original[dataset_name][0]

    def get_obs_pred_mean(self, dataset_name):
        return self.obs_pred_mean[dataset_name][0]

    def get_obs_pred_variance(self, dataset_name):
        return self.obs_pred_variance[dataset_name][0]

    def get_obs_name(self, dataset_name):
        return self.obs_name[dataset_name]

    def get_dep_variable_name(self, dataset_name):
        return self.dep_variable_name[dataset_name]

    def get_dep_variable_values(self, dataset_name):
        return self.dep_variable_values[dataset_name]

    def get_chi2pp_validation(self):
        return self.chi2pp_validation

    def get_chi2pp_kfold(self):
        return self.chi2pp_kfold

    def get_full_pdf(self, pdf_name, flav_name, x, k2):
        return self.pdf_pred[pdf_name]["full_pdf"][flav_name].ev(x, k2)

    def get_coll_part(self, pdf_name, flav_name, x):
        return self.pdf_pred[pdf_name]["coll_part"][flav_name](x)

    def get_transverse_part(self, pdf_name, flav_name, k2):
        return self.pdf_pred[pdf_name]["branch_k"][flav_name](k2)

    def get_full_pdf_rep(self, rnum, pdf_name, flav_name, x, k2):
        return self.rep_fitters[rnum].get_full_pdf(pdf_name, flav_name, x, k2)

    def get_coll_part_rep(self, rnum, pdf_name, flav_name, x):
        return self.rep_fitters[rnum].get_coll_part(pdf_name, flav_name, x)

    def get_transverse_part_rep(self, rnum, pdf_name, flav_name, k2):
        return self.rep_fitters[rnum].get_transverse_part(pdf_name, flav_name, k2)

    def get_nrep(self):
        return self.nrep
