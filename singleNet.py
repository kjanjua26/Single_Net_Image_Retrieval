from __future__ import print_function, absolute_import, division # the handle the diff between py2, py3
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from gensim.models import KeyedVectors # to load google's w2v model
from gensim.models import Word2Vec as w2v 
import extract_features

sentences_fileName = ''
images_fileName = ''

sample_sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

sentence_embedding, _ = extract_features.get_glove_model(sample_sentences) # returns words not in vocab as well. 
w2v_emb = extract_features.get_w2v_model(sample_sentences)
print("this", w2v_emb[0])
print("this", sentence_embedding[0])
