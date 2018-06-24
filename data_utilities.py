import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math, glob
from gensim.models.doc2vec import TaggedDocument
import extract_features

sent_path = 'flicker-sample/sentences-flicker.txt' # a text file containing sentences
img_path = 'flicker-sample/images'
sent_file = open(sent_path, 'r').readlines()

def _get_sentences(img_name):	
	img_desc = []
	for i in sent_file:
		if i.find(img_name) == -1:
			pass
		else:
			sent = ' '.join(i.split()[1:len(i.split())-1]) 
			img_desc.append(TaggedDocument(words=sent.split(), tags=[img_name]))
	return img_desc

def _get_training_data_corr():
	correct_list = []
	for i in glob.glob(img_path+"/*.jpg"):
		i = i.split("/", 2)
		name = i[2].replace(".jpg","")
		img_name = "flicker-sample/images/" + i[2]
		img_ft = extract_features.get_img_features_vgg16(img_name)
		print "For {0}".format(img_name)
		sent_list = _get_sentences(name)
		sent_fts = extract_features.get_doc2v_model(sent_list)
		print "Sentence Feature: ", type(sent_fts), sent_fts.shape
		print "Image Features: ", type(img_ft), img_ft.shape
		img_ft_reshaped = np.reshape(img_ft, (-1, 2))
		sent_ft_reshaped = np.reshape(sent_fts, (-1,2))
		print "Reshaped Sent Feature: ", sent_ft_reshaped.shape
		print "Reshaped Img Feature: ", img_ft_reshaped.shape
		corr_fts = np.concatenate((img_ft_reshaped, sent_ft_reshaped), axis=0)
		print "Corr_Fts Shape: ", corr_fts.shape
		correct_list.append(corr_fts)
		print ""
	correct_list_arr = np.asarray(correct_list)
	print "Shape of correct_list Array: ", correct_list_arr.shape
	return correct_list_arr # returned feature array. 
