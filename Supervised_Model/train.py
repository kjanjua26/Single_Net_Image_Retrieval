from __future__ import print_function, absolute_import, division # the handle the diff between py2, py3
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math, os
import extract_features
import network
import data_utilities
# Using the flickr30k dataset 

sentences_fileName = ''
images_fileName = ''
num_steps = 500
BATCH_SIZE = 2
steps_per_epoch = 100
init_learning_rate = 0.0001
num_train_samples = 31783
restore_path = 'models/'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

img_fts = tf.placeholder(tf.float32, shape=[None, 2])
sent_fts = tf.placeholder(tf.float32, shape=[None, 2])
labels = tf.placeholder(tf.bool, shape=[None, BATCH_SIZE])

if os.path.isfile("wrong_fts.npy") and os.path.isfile("corr_fts.npy"):
	corr_fts_train = np.load('corr_fts.npy')
	wrong_fts_train = np.load('wrong_fts.npy')
else:
	corr_fts_train = data_utilities._get_training_data_corr()
	wrong_fts_train = data_utilities._get_training_data_wrong()

# train the network 
def train():
	loss = network.compute_loss(correct_X, wrong_X)
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, steps_per_epoch, 0.794, staircase=True)
	optim = tf.train.AdamOptimizer(init_learning_rate)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = optim.minimize(loss, global_step=global_step)
		saver = tf.train.Saver(save_relative_paths=True)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_steps):
			for start, end in zip(range(0, num_train_samples, BATCH_SIZE),range(BATCH_SIZE, num_train_samples + 1,BATCH_SIZE)): # batches
				feed_dict = {
				    correct_X: corr_fts_train[start:end],
				    wrong_X: wrong_fts_train[start:end]
				}
				_, loss_Val = sess.run([train_step, loss], feed_dict=feed_dict)
				print('Epoch: %d Step: %d Loss: %f' % (i // steps_per_epoch, i, loss_Val))
				if i % steps_per_epoch == 0 and i > 0:
				    print('Saving checkpoint at step %d' % i)
				    #saver.save(sess, FLAGS.save_dir, global_step = global_step)


if __name__ == '__main__':
	train()
