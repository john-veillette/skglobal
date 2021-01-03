from scikeras.wrappers import KerasClassifier 
from tensorflow.compat.v1.keras.regularizers import L1L2
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras import Input
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.backend import reshape, get_session
import tensorflow.python.util.deprecation as deprecation
import tensorflow.compat.v1 as tf
from mne.decoding.base import LinearModel
from warnings import filterwarnings, catch_warnings
import global_objectives as go
import numpy as np

tf.disable_v2_behavior()

LOSS_OPTIONS = {
    'precision_recall_auc': go.precision_recall_auc_loss,
    'roc_auc': go.roc_auc_loss,
    'recall_at_precision': go.recall_at_precision_loss,
    'precision_at_recall': go.precision_at_recall_loss,
    'false_positive_rate_at_true_positive_rate': go.false_positive_rate_at_true_positive_rate_loss,
    'true_positive_rate_at_false_positive_rate': go.true_positive_rate_at_false_positive_rate_loss
    }

def get_loss(loss, **loss_args):
    '''
    Returns a Keras-compatible loss function given specification
    '''
    if loss in LOSS_OPTIONS:
        loss_func = LOSS_OPTIONS[loss]
    else:
        raise ValueError('Specified loss not supported by this estimator!')

    def keras_loss(y_true, y_pred):
        y_true = reshape(y_true, (-1, 1)) 
        y_pred = reshape(y_pred, (-1, 1))  
        return loss_func(y_true, y_pred, **loss_args)[0]

    return keras_loss


class LinearClassifier(LinearModel):
    '''
    Binary classifier that optimizes global objective functions
    '''

    def __init__(self, loss, penalty = 'l2', C = 1., tol = 1e-4, 
        max_iter = 1000, batch_size = None, warnings = False, **loss_args):

        self.warnings_ = warnings
        if not warnings:
            deprecation._PRINT_DEPRECATION_WARNINGS = False

        if penalty is 'l1':
            c1 = C 
            c2 = 0.
        elif penalty is 'l2':
            c1 = 0.
            c2 = C 
        elif penalty is not 'none':
            raise ValueError("Supported penalties are 'l1', 'l2', and 'none'.")

        def build_model(meta):
            # get info about data dimensions
            X_shape_ = meta["X_shape_"]
            n_classes_ = meta["n_classes_"]
            # and build model
            mod = Sequential()
            mod.add(Input(shape = X_shape_[1:]))
            mod.add(
                Dense(
                    1, 
                    activation = 'sigmoid', 
                    kernel_regularizer = L1L2(l1 = c1, l2 = c2),
                    name = 'proba'
                    )
                )
            mod.compile(
                loss = get_loss(loss, **loss_args), 
                optimizer = 'sgd',
                metrics = ['accuracy']
                )
            # keras will forget to compile some variables in
            # the custom tf loss functions, so we fix that here
            init_op = tf.global_variables_initializer()
            sess = get_session()
            sess.run(init_op)
            return mod 

        if warnings:
            model = KerasClassifier(build_model, epochs = max_iter)
        else:
            with catch_warnings():
                filterwarnings("ignore")
                model = KerasClassifier(build_model, epochs = max_iter)
        self.model = model 
        self._estimator_type = getattr(model, "_estimator_type", None)  

        # save training paramters to use later
        self.batch_size = batch_size
        self.stopping_criteria = Stop(tol)
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight = None):
        if self.warnings_:
            return self._fit(X, y, sample_weight = None)
        else:
            with catch_warnings():
                filterwarnings("ignore")
                return self._fit(X, y, sample_weight = None)

    def _fit(self, X, y, sample_weight = None):
        if np.unique(y).shape[0] > 2:
            raise ValueError(
                "Global loss functions currently only support binary "
                "classification but more than two class labels were given."
                )
        fit_params = {}
        if self.batch_size is not None:
            fit_params['batch_size'] = self.batch_size
        else: # use non-stochastic gradient descent 
            fit_params['batch_size'] = y.shape[0]
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        fit_params['callbacks'] = [self.stopping_criteria]
        fit_params['verbose'] = 0
        return super().fit(X, y, **fit_params)

    @property
    def coef_(self):
        '''
        an (n_features,) array for positive class
        '''
        model = self.model.model_ # the exposed keras model
        layer = model.get_layer(name = 'proba')
        weights = layer.get_weights()[0]
        # bias = layer.get_weights()[1][0]
        return np.array([w[0] for w in weights])

    @property
    def bias(self):
        model = self.model.model_ # the exposed keras model
        layer = model.get_layer(name = 'proba')
        return layer.get_weights()[1][0]    

    @property
    def filters_(self):
        return self.coef_

    @property
    def classes_(self):
        return self.model.classes_

    @property
    def n_classes_(self):
        return self.model.n_classes_

class Stop(EarlyStopping):
    '''
    Since the global objective losses change over time to enforce certain 
    contstraints, training loss isn't strictly decreasing even if the model
    is training well. So we need to extend EarlyStopping to stop when the
    training loss converges (absolute change less than tolerance) rather than 
    when it stops decreasing. It can still return the weights to the minimum
    loss though, if you set restore_best_weights = True. Please carefully 
    consider the time-evolving behavior of your loss function before doing so.
    '''

    def __init__(self, tol, restore_best_weights = False):
        super().__init__(monitor = 'loss', 
            min_delta = tol,
            restore_best_weights = restore_best_weights,
            patience = 10,
            mode = 'min'
            )
        self.tol = tol 
        self.first_epoch = True

    def on_epoch_end(self, epoch, logs = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.first_epoch:
            self.first_epoch = False 
            self.last_epoch = current
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()            
        if np.abs(current - self.last_epoch) < self.tol:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print(
                            "Restoring model weights from the end " 
                            "of the best epoch."
                            )
                    self.model.set_weights(self.best_weights)
        else:
            self.wait = 0
        self.last_epoch = current

 




















