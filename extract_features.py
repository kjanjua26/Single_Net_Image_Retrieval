from __future__ import print_function, absolute_import, division # the handle the diff between py2, py3
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from gensim.models import Word2Vec as w2v 
from gensim.models.doc2vec import TaggedDocument
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec as d2v
import nltk
import numpy as np

def tokenize_sentences(img_desc):
	tokens = []
	for i in img_desc:
		for j in i:
			tokens.append(nltk.word_tokenize(j))
	return tokens

def get_img_features_vgg16(img_path):
	model = VGG16(weights='imagenet', include_top=False) # to extract features 
	print("Loaded VGG16 model loaded successfully.")
	img = image.load_img(img_path)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	features = np.array(features)
	return features.flatten() # img features returned

def get_w2v_model(sentences):
	model = w2v(sentences, min_count=1, workers=4)
	print("Word2Vec Model Loaded Successfully.")
	features = model[model.wv.vocab]
	return features

def get_doc2v_model(sentences): # doc2vec code
	model = d2v(vector_size=100, min_count=5, workers=5)
	model.build_vocab(sentences)
	print("Doc2Vec Model Loaded Successfully.")
	return  model.docvecs[0]

def get_glove_model(sentences):
	emb_ft = []
	not_in_vocab = []
	filename = 'glove.6B/glove.6B.100d.txt.word2vec'
	'''
		A one time process. 

		glove_input_file = 'glove.6B/glove.6B.100d.txt'
		word2vec_output_file = 'glove.6B/glove.6B.100d.txt.word2vec'
		glove2word2vec(glove_input_file, word2vec_output_file)
	'''
	model = KeyedVectors.load_word2vec_format(filename, binary=False)
	print("Stanford Glove 100d Model Loaded Successfully.")
	for i in sentences:
		for j in i:
			try:
				emb_ft.append(model[j])
			except:
				not_in_vocab.append(j)
	return emb_ft, not_in_vocab
