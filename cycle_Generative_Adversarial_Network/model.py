import tensorflow as tf
from unet_module import *

class cycleGAN:
    def __init__(self):
        self.input_A = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_A')          # self.input_A = X
        self.input_B = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_B')          # self.input_B = Y

        self.Generator_A2B, self.Generator_A2B_params = self.GenA2B(self.input_A, False)        # self.Generator_A2B = G(X) -> hat_Y
        self.Generator_B2A, self.Generator_B2A_params = self.GenB2A(self.input_B, False)        # self.Generator_B2A = F(Y) -> hat_X
        
        self.Generator_A2B2A, self.Generator_A2B2A_params = self.GenB2A(self.Generator_A2B, True) # self.Generator_A2B2A = F(G(X)) -> hat_hat_X
        self.Generator_B2A2B, self.Generator_B2A2B_params = self.GenA2B(self.Generator_B2A, True) # self.Generator_B2A2B = G(F(Y)) -> hat_hat_Y

        self.discriminator_A, self.discriminator_A_params = self.DesA(self.input_A, False)      # self.discriminator_A = Dx(X)
        self.discriminator_B, self.discriminator_B_params = self.DesB(self.input_B, False)      # self.discriminator_B = Dy(Y) 

        self.discriminator_AB, self.discriminator_AB_params = self.DesA(self.Generator_B2A, True)   # self.discriminator_AB = Dx(F(Y))
        self.discriminator_BA, self.discriminator_BA_params = self.DesB(self.Generator_A2B, True)   # self.discriminator_BA = Dy(G(X))

        cycle_consisty_loss_A = tf.reduce_mean(tf.abs(self.input_A - self.Generator_A2B2A))     # || X - F(G(X)) || -> 얼마나 X를 재생성해내느냐
        cycle_consisty_loss_B = tf.reduce_mean(tf.abs(self.input_B - self.Generator_B2A2B))     # || Y - G(F(Y)) || -> 얼마나 Y를 재생성해내느냐

    def GenA2B(self, input, reuse):
        with tf.variable_scope('GenA2B') as scope:
            if reuse:
                scope.reuse_variables()
            hidden_1 = encoder(input, 3, 9, False, name='hidden_1')
            hidden_2 = encoder(hidden_1, 9, 36, True, name='hidden_2')
            hidden_3 = encoder(hidden_2, 36, 144, True, name='hidden_3')
            hidden_4 = encoder(hidden_3, 144, 576, True, name='hidden_4')
            hidden_5 = decoder(hidden_3, hidden_4, 144, 576, 144, name='hidden_5')
            hidden_6 = decoder(hidden_2, hidden_5, 36, 144, 36, name='hidden_6')
            output = decoder(hidden_1, hidden_6, 9, 36, 3, name='output')
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GenA2B')
        return output, params

    def GenB2A(self, input, reuse):
        with tf.variable_scope('GenB2A') as scope:
            if reuse:
                scope.reuse_variables()
            hidden_1 = encoder(input, 3, 9, False, name='hidden_1')
            hidden_2 = encoder(hidden_1, 9, 36, True, name='hidden_2')
            hidden_3 = encoder(hidden_2, 36, 144, True, name='hidden_3')
            hidden_4 = encoder(hidden_3, 144, 576, True, name='hidden_4')
            hidden_5 = decoder(hidden_3, hidden_4, 144, 576, 144, name='hidden_5')
            hidden_6 = decoder(hidden_2, hidden_5, 36, 144, 36, name='hidden_6')
            output = decoder(hidden_1, hidden_6, 9, 36, 3, name='output')
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GenB2A')
        return output, params
    
    def DesA(self, input, reuse):
        with tf.variable_scope('DesA') as scope:
            if reuse:
                scope.reuse_variables()
            hidden_1 = tf.layers.conv2d(inputs=input, filters=9, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_1')
            max_pool_1 = tf.layers.max_pooling2d(inputs=hidden_1, pool_size=[2, 2], strides=[2, 2], name='max_pool_1')
            hidden_2 = tf.layers.conv2d(inputs=max_pool_1, filters=36, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_2')
            max_pool_2 = tf.layers.max_pooling2d(inputs=hidden_2, pool_size=[2, 2], strides=[2, 2], name='max_pool_2')
            hidden_3 = tf.layers.conv2d(inputs=max_pool_2, filters=9, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_3')
            max_pool_3 = tf.layers.max_pooling2d(inputs=hidden_3, pool_size=[2, 2], strides=[2, 2], name='max_pool_3')
            hidden_4 = tf.layers.conv2d(inputs=max_pool_3, filters=36, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_4')
            max_pool_4 = tf.layers.max_pooling2d(inputs=hidden_4, pool_size=[2, 2], strides=[2, 2], name='max_pool_4')
            flatten = tf.layers.flatten(max_pool_4, name='flatten')
            hidden_5 = tf.layers.dense(inputs=flatten, units=16*36, activation=tf.nn.relu, name='hidden_5')
            hidden_6 = tf.layers.dense(inputs=hidden_5, units=36, activation=tf.nn.relu, name='hidden_6')
            output = tf.layers.dense(inputs=hidden_6, units=1, activation=tf.nn.sigmoid, name='output')
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DesA')
        return output, params
    
    def DesB(self, input, reuse):
        with tf.variable_scope('DesB') as scope:
            if reuse:
                scope.reuse_variables()
            hidden_1 = tf.layers.conv2d(inputs=input, filters=9, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_1')
            max_pool_1 = tf.layers.max_pooling2d(inputs=hidden_1, pool_size=[2, 2], strides=[2, 2], name='max_pool_1')
            hidden_2 = tf.layers.conv2d(inputs=max_pool_1, filters=36, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_2')
            max_pool_2 = tf.layers.max_pooling2d(inputs=hidden_2, pool_size=[2, 2], strides=[2, 2], name='max_pool_2')
            hidden_3 = tf.layers.conv2d(inputs=max_pool_2, filters=9, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_3')
            max_pool_3 = tf.layers.max_pooling2d(inputs=hidden_3, pool_size=[2, 2], strides=[2, 2], name='max_pool_3')
            hidden_4 = tf.layers.conv2d(inputs=max_pool_3, filters=36, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='hidden_4')
            max_pool_4 = tf.layers.max_pooling2d(inputs=hidden_4, pool_size=[2, 2], strides=[2, 2], name='max_pool_4')
            flatten = tf.layers.flatten(max_pool_4, name='flatten')
            hidden_5 = tf.layers.dense(inputs=flatten, units=16*36, activation=tf.nn.relu, name='hidden_5')
            hidden_6 = tf.layers.dense(inputs=hidden_5, units=36, activation=tf.nn.relu, name='hidden_6')
            output = tf.layers.dense(inputs=hidden_6, units=1, activation=tf.nn.sigmoid, name='output')
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DesB')
        return output, params
    

cycle = cycleGAN()