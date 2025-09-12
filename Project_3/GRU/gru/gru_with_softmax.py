# standards import 
import warnings

# third party import
from sklearn import preprocessing
import tensorflow as tf

# custom/local import
import config_manager as cm
import builder as build

# ignoring warning
warnings.filterwarnings("ignore", category=UserWarning)


class GRUWithSoftmax():
    """
        GRU Model with Softmax Output
    """
    
    def __init__(self):
        self.config = cm.gru_config_experiment_1
        self.builder = build.GRUBuilder()
        self.model = self.builder.model
        
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=self.config.epochs,
                       batch_size=self.config.batch_size)
        
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)