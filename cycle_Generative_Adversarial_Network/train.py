import tensorflow as tf
from unet_module import *
from model import cycleGAN
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random

sess = tf.Session()
cycle = cycleGAN()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('initializer')

SAVE_PATH = './save'
MODEL_NAME = 'test'

A = []
B = []
for root, dirs, files in os.walk('mini/trainA'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        img = mpimg.imread(full_fname) / 255
        A.append(img)

for root, dirs, files in os.walk('mini/trainB'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        img = mpimg.imread(full_fname) / 255
        B.append(img)
print('Read dataset')



for epoch in range(100):
    random.shuffle(A)
    random.shuffle(B)
    _, _, gx_loss, gy_loss = sess.run([cycle._gx_train_step, cycle._gy_train_step, cycle._gx_loss, cycle._gy_loss], feed_dict={cycle.input_A: A, cycle.input_B: B})
    _, _, dx_loss, dy_loss = sess.run([cycle._dx_train_step, cycle._dy_train_step, cycle._dx_loss, cycle._dy_loss], feed_dict={cycle.input_A: A, cycle.input_B: B})
    print('epoch:', epoch, 'gx_loss:', gx_loss, 'gy_loss:', gy_loss, 'dx_loss:', dx_loss, 'dy_loss:', dy_loss)
    result_A, result_B = sess.run([cycle._gy, cycle._gx], feed_dict={cycle.input_A: A, cycle.input_B: B})
    path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
    print("saved at {}".format(path))
