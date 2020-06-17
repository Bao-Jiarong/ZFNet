'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-14
  email        : bao.salirong@gmail.com
  Task         : ZFNet Implementation
  Dataset      : MNIST Digits (0,1,...,9)
'''

import tensorflow as tf

class Block(tf.keras.models.Sequential):
    def __init__(self,n,kernel_size,stride=(1,1),padding='same'):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(filters = n, kernel_size=kernel_size,strides=stride,padding = padding,activation = "relu"))

        self.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))


class ZFNet(tf.keras.models.Sequential):
    def __init__(self, input_shape, classes,filters = 32):
        super().__init__()
        self.add(tf.keras.layers.InputLayer(input_shape = input_shape))

        # Backbone
        self.add(Block(n = filters * 3, kernel_size=(7,7), stride=(2,2), padding='valid'))
        self.add(Block(n = filters * 8, kernel_size=(5,5), stride=(2,2)))
        self.add(tf.keras.layers.Conv2D(filters = filters * 16, kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu"))
        self.add(tf.keras.layers.Conv2D(filters = filters * 32, kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu"))
        self.add(Block(n = filters * 16 ,kernel_size=(3,3)))

        # top
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units = filters * 288, activation = "relu"))
        self.add(tf.keras.layers.Dense(units = filters * 128, activation = "relu"))
        self.add(tf.keras.layers.Dense(units = filters * 128, activation = "relu"))
        self.add(tf.keras.layers.Dense(units = classes,activation = "softmax"))
