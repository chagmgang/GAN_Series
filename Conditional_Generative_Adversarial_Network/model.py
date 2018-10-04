import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

class GAN:
    def __init__(self):        
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.noise = tf.placeholder(tf.float32, [None, 784])
        self.labels = tf.placeholder(tf.float32,[None, 10])

        self.gen, self.params_gen = self.Generator(self.noise, self.labels)
        self.D_gene, self.param_2 = self.Discriminator(self.gen, self.labels, False)
        self.D_real, self.param_1 = self.Discriminator(self.input, self.labels, True)

        self.loss_D = tf.reduce_mean(tf.log(self.D_real + 1e-10) + tf.log(1 - self.D_gene + 1e-10))
        self.loss_G = tf.reduce_mean(tf.log(self.D_gene + 1e-10))

        self.train_D = tf.train.AdamOptimizer(0.0005).minimize(-self.loss_D,
                                                            var_list=self.param_1)
        self.train_G = tf.train.AdamOptimizer(0.0005).minimize(-self.loss_G,
                                                            var_list=self.params_gen)

    def Discriminator(self, input, labels, reuse):
        with tf.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()
            concat = tf.concat([input, labels], axis=1)
            hidden = tf.layers.dense(inputs=concat, units=256, activation=tf.nn.relu, name='hidden_1')
            output = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid, name='output')
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        return output, params

    def Generator(self, input, labels):
        with tf.variable_scope('gen'):
            concat = tf.concat([input, labels], axis=1)
            hidden = tf.layers.dense(inputs=concat, units=256, activation=tf.nn.relu)
            output = tf.layers.dense(inputs=hidden, units=784, activation=tf.nn.relu)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        return output, params

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

sess = tf.Session()
gan = GAN()
sess.run(tf.global_variables_initializer())
batch_size = 500
total_batch = int(mnist.train.num_examples/batch_size)
total_epoch = 300
n_noise = 784

loss_val_D, loss_val_G = 0, 0
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)
        
        _, loss_val_D = sess.run([gan.train_D, gan.loss_D], feed_dict={gan.input: batch_xs, gan.noise: noise, gan.labels: batch_ys})
        _, loss_val_G = sess.run([gan.train_G, gan.loss_G], feed_dict={gan.noise: noise, gan.labels: batch_ys})
        _, loss_val_G = sess.run([gan.train_G, gan.loss_G], feed_dict={gan.noise: noise, gan.labels: batch_ys})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        labels = np.eye(10)
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(gan.gen, feed_dict={gan.noise: noise, gan.labels: labels})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('./Conditional_Generative_Adversarial_Network/src/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)