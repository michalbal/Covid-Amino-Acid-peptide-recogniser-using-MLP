import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Embedding
from tensorflow.keras import Model


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.f = Flatten()
    self.d0 = Dense(128, activation='relu')
    self.d1 = Dense(64, activation='sigmoid')
    self.drop1 = Dropout(0.4)
    self.d2 = Dense(32, activation='relu', kernel_regularizer='l2')
    self.drop2 = Dropout(0.2)
    self.d3 = Dense(2, activation='sigmoid')

  def call(self, x):
    x = self.f(x)
    x = self.d0(x)
    x = self.d1(x)
    x = self.drop1(x)
    x = self.d2(x)
    x = self.drop2(x)
    return self.d3(x)