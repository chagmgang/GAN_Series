import tensorflow as tf
from unet_module import *

class cycleGAN:
    def __init__(self):
        # GenA2B = Gx->y, GenB2A = Gy->x, DesA = Dx, DesB = Dy, self.input_A = X, self.input_B = Y
        self.lambda_ = 0.9
        
        self.input_A = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_A')
        self.input_B = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_B')

        self._gy, self.Gy = self.GenA2B(self.input_A, reuse=False)              # self._gy = Gx->y(X), self.Gy = parameters of Gx->y
        self._gx, self.Gx = self.GenB2A(self.input_B, reuse=False)              # self._gx = Gy->x(Y), self.Gx = parameters of Gy->x

        self._gy_from_gx, self.Gy = self.GenA2B(self._gx, reuse=True)           # self._gy_from_gx = Gx->y(Gy->x(Y))
        self._gx_from_gy, self.Gx = self.GenB2A(self._gy, reuse=True)           # self._gx_from_gy = Gy->x(Gx->y(X))

        self._real_dx, self.Dx = self.DesA(self.input_A, reuse=False)           # self._real_dx = Dx(X), self.Dx = parameters of Dx
        self._fake_dx, self.Dx = self.DesA(self._gx, reuse=True)                # self._fake_dx = Dx(Gy->x(Y))
        self._fake_dx_g, self.Dx = self.DesA(self._gx_from_gy, reuse=True)      # self._fake_dx_g = Dx(Gy->x(Gx->y(X)))

        self._real_dy, self.Dy = self.DesB(self.input_B, reuse=False)           # self._real_dy = Dy(Y), self.Dy = parameters of Dy
        self._fake_dy, self.Dy = self.DesB(self._gy, reuse=True)                # self._fake_dy = Dy(Gx->y(X))
        self._fake_dy_g, self.Dy = self.DesB(self._gy_from_gx, reuse=True)      # self._fake_dy_g = Dy(Gx->y(Gy->x(Y)))

        cycle_loss = tf.reduce_mean(tf.abs(self._gx_from_gy - self.input_A) + tf.abs(self._gy_from_gx - self.input_B))  # || Gy->x(Gx->y(X)) - X || + || Gx->y(Gy->x(Y)) - Y ||
        
        self._gx_loss =  0.5 * tf.reduce_mean(tf.square(self._fake_dx_g - 1.)) + cycle_loss
        self._gy_loss =  0.5 * tf.reduce_mean(tf.square(self._fake_dy_g - 1.)) + cycle_loss

        self._dx_loss =  0.5 * tf.reduce_mean(tf.square(self._real_dx - 1.)) + 0.5 * tf.reduce_mean(tf.square(self._fake_dx))
        self._dy_loss =  0.5 * tf.reduce_mean(tf.square(self._real_dy - 1.)) + 0.5 * tf.reduce_mean(tf.square(self._fake_dy))

        self._gx_train_step = tf.train.AdamOptimizer(0.001).minimize(self._gx_loss, var_list=self.Gy)
        self._gy_train_step = tf.train.AdamOptimizer(0.001).minimize(self._gy_loss, var_list=self.Gx)
        self._dx_train_step = tf.train.AdamOptimizer(0.001).minimize(self._dx_loss, var_list=self.Dx)
        self._dy_train_step = tf.train.AdamOptimizer(0.001).minimize(self._dy_loss, var_list=self.Dy)

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
    

