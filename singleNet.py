from __future__ import print_function, absolute_import, division # the handle the diff between py2, py3
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math
import extract_features
import nltk
# Using the flickr30k dataset 

sentences_fileName = ''
images_fileName = ''

correct_X = tf.placeholder(tf.float32, [None, 2])
wrong_X = tf.placeholder(tf.float32, [None, 2])

sample_sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

img_dec = [['A person wearing a black jacket is bending over a table'],
		['An old person is wearing a black jacket'], 
		['An old person in black jacket is trying to pick up something']]
img = 'sampleImages/1.jpg'

def tokenize_sentences(img_desc):
	tokens = []
	for i in img_desc:
		for j in i:
			tokens.append(nltk.word_tokenize(j))
	return tokens 

tokens = tokenize_sentences(img_dec)

img_features = extract_features.get_img_features_vgg16(img)
sent_features = extract_features.get_w2v_model(tokens)

print("Reshaping the features!")
print("Image Shape: ", img_features.shape)
print("Sentence Shape: ", sent_features.shape)
img_features_2d = np.reshape(img_features, (-1, 2)) # converting to 2d feature array
sent_features_2d = np.reshape(sent_features, (-1, 2))
print("Img size: ", img_features.size)
print("sent_features_2d shape: ", sent_features_2d.shape)
print("2D Image Shape: ", img_features_2d.shape)

correct_fts = np.concatenate((img_features_2d, sent_features_2d), axis=0)
print("Total Shape: ", correct_fts.shape)
