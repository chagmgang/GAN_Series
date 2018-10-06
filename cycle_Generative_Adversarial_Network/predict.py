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
saver = tf.train.Saver()

SAVE_PATH = './save'
MODEL_NAME = 'test'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
saver.restore(sess, checkpoint)
print('initializer')

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

result_A, result_B = sess.run([cycle._gy, cycle._gx], feed_dict={cycle.input_A: A, cycle.input_B: B})

plt.imshow(A[0])
plt.savefig('input_A_'+str(0)+'.png')
plt.imshow(result_A[0])
plt.savefig('output_A_'+str(0)+'.png')

plt.imshow(B[0])
plt.savefig('input_B_'+str(0)+'.png')
plt.imshow(result_B[0])
plt.savefig('output_B_'+str(0)+'.png')

print(result_A.shape)
