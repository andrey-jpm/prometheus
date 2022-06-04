import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class DataKFold():
    def __init__(self, n_kfolds, validation_size):
        self.n_kfolds = n_kfolds
        self.validation_size = validation_size

    def get_masks(self, exp_data):
        training_mask = [[] for i in range(self.n_kfolds)]
        validation_mask = [[] for i in range(self.n_kfolds)]
        kfold_mask = [[] for i in range(self.n_kfolds)]
        for exp_dict in exp_data:
            values = exp_dict["exp_value"]
            if self.n_kfolds == 1:
                kfold_mask[0] += [[0 for i in range(len(values))]]
                values_train, values_validation = train_test_split(values, test_size=self.validation_size)
                training_mask[0] += [[int(val in values_train) for val in values]]
                validation_mask[0] += [[int(val in values_validation) for val in values]]
            else:
                kf = KFold(n_splits=self.n_kfolds,shuffle=True)
                kfold_num = 0
                for train_test_index, kfold_index in kf.split(values):
                    train_index, validation_index = train_test_split(train_test_index, test_size=self.validation_size)
                    kfold_mask[kfold_num] += [[int(index in kfold_index) for index in range(len(values))]]
                    training_mask[kfold_num] += [[int(index in train_index) for index in range(len(values))]]
                    validation_mask[kfold_num] += [[int(index in validation_index) for index in range(len(values))]]

                    kfold_num += 1

        return training_mask, validation_mask, kfold_mask