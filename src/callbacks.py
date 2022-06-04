from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from config import MAX_LAMBDA, POS_THRESHOLD, POS_DENSITY

class ValidationCheckCallback(Callback):
    def __init__(self, model, input_tftensor, max_epoch, stopping_patience):
        self._model = model
        self._input_tftensor = input_tftensor

        self.prediction_history = []
        self.chi2_validation_history = []
        self.chi2_training_history = []

        self.best_epoch = 0
        self._best_weights = self._model.get_weights()
        self.chi2_validation_best = self._model(self._input_tftensor, training=False)["chi2_validation"].numpy()[0]

        self.max_epoch = max_epoch
        self.stopping_epoch = int(self.max_epoch * stopping_patience)

        self.stopping_epoch_count = 0

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        prediction = self._model(self._input_tftensor, training=False)
        positivity = K.sum(prediction["pos_dataset"], axis=-1).numpy()[0] / POS_DENSITY
        self.prediction_history += [prediction]
        self.chi2_validation_history += [prediction["chi2_validation"].numpy()[0]]
        self.chi2_training_history += [prediction["chi2_training"].numpy()[0]]

        if (self.chi2_validation_history[-1] < self.chi2_validation_best) & (positivity < POS_THRESHOLD):
            self.best_epoch = epoch
            self._best_weights = self._model.get_weights()
            self.chi2_validation_best = self.chi2_validation_history[-1]
            self.stopping_epoch_count = 0
        else:
            self.stopping_epoch_count += 1

        if self.stopping_epoch_count > self.stopping_epoch:
            self._model.set_weights(self._best_weights)
            self.model.stop_training = True

class UpdateMultipliersCallback(Callback):
    def __init__(self, max_epochs, lambda_update_freq):
        self.max_epochs = max_epochs
        self.lambda_update_freq = lambda_update_freq
        self.steps = int(self.max_epochs/self.lambda_update_freq)
        self.multiplier = pow(MAX_LAMBDA, 1.0/self.steps)
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.lambda_update_freq == 0:
            # get weights of the LM layer
            weights = self.model.get_layer("LMLayer").get_weights()[0]
            # update weights of the LM layer
            self.model.get_layer("LMLayer").set_weights([weights * self.multiplier])
