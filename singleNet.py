from __future__ import print_function, absolute_import, division # the handle the diff between py2, py3
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math
import extract_features
import network
# Using the flickr30k dataset 

sentences_fileName = ''
images_fileName = ''
num_steps = 500
steps_per_epoch = 100
init_learning_rate = 0.0001
restore_path = 'models/'

correct_X = tf.placeholder(tf.float32, shape=[None, 2])
wrong_X = tf.placeholder(tf.float32, shape=[None, 2])

# temporary data 
sample_sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

img_dec = [['A person wearing a black jacket is bending over a table'],
		['An old person is wearing a black jacket'], 
		['An old person in black jacket is trying to pick up something']]
img = 'sampleImages/1.jpg'

tokens = extract_features.tokenize_sentences(img_dec)

img_features = extract_features.get_img_features_vgg16(img)
sent_features = extract_features.get_w2v_model(tokens)
sent_fts_wr = extract_features.get_w2v_model(sample_sentences)

print("Reshaping the features!")
print("Image Shape: ", img_features.shape)
print("Sentence Shape: ", sent_features.shape)
img_features_2d = np.reshape(img_features, (-1, 2)) # converting to 2d feature array
sent_features_2d = np.reshape(sent_features, (-1, 2))
sent_ft_wr_2d = np.reshape(sent_fts_wr, (-1, 2))
print("Img size: ", img_features.size)
print("sent_features_2d shape: ", sent_features_2d.shape)
print("2D Image Shape: ", img_features_2d.shape)
print("sent_ft_wr_2d shape: ",sent_ft_wr_2d.shape)
correct_fts = np.concatenate((img_features_2d, sent_features_2d), axis=0)
wrong_fts = np.concatenate((img_features_2d, sent_ft_wr_2d), axis=0)
print("Total Shape: ", correct_fts.shape)


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
		#if restore_path:	
		#    print('restoring checkpoint', restore_path)
		#    saver.restore(sess, restore_path.replace('.meta', ''))
		#    print('done')
		for i in range(num_steps):
			feed_dict = {
			    correct_X: correct_fts,
			    wrong_X: wrong_fts
			}
			_, loss_Val = sess.run([train_step, loss], feed_dict=feed_dict)
			if i % 10 == 0: 
				print('Epoch: %d Step: %d Loss: %f' % (i // steps_per_epoch, i, loss_Val))
			if i % steps_per_epoch == 0 and i > 0:
			    print('Saving checkpoint at step %d' % i)
			    saver.save(sess, FLAGS.save_dir, global_step = global_step)


if __name__ == '__main__':
	train()
