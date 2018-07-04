import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math, glob
from gensim.models.doc2vec import TaggedDocument
import extract_features
import random

sent_path = '/home/super/datasets/flicker30/results_20130124_orig.token' # a text file containing sentences
img_path = '/home/super/datasets/flicker30/images/train/flickr30k_100images'
sent_file = open(sent_path, 'r').readlines()

def get_img_fts():
	img_fts_list = []
	for i in glob.glob(img_path+"/*.jpg"):
		count += 1
		i_ed = i.split("/", 8)
		name = i_ed[8].replace(".jpg","")
		img_name = i
		img_ft = extract_features.get_img_features_vgg16(img_name)
		print("For Count: {0}, {1}".format(count,img_name))
		print("Image Features: ", type(img_ft), img_ft.shape)
		img_ft_reshaped = np.reshape(img_ft, (-1, 2))
		print("Reshaped Img Feature: ", img_ft_reshaped.shape)
		img_fts_list.append(img_ft_reshaped)
		print("")
	img_fts_arr = np.concatenate(img_fts_list, axis=0)
	np.save('img_fts_arr.npy', img_fts_arr)
	return img_fts_arr

def get_sent_fts():
	sent_fts_list = []
	for i in glob.glob(img_path+"/*.jpg"):
		count += 1
		i_ed = i.split("/", 8)
		name = i_ed[8].replace(".jpg","")
		img_name = i
		print("For Count: {0}, {1}".format(count,img_name))
		sent_list = _get_corr_sentences(name)
		sent_fts = extract_features.get_doc2v_model(sent_list)
		print("Sentence Feature: ", type(sent_fts), sent_fts.shape)
		sent_ft_reshaped = np.reshape(sent_fts, (-1,2))
		print("Reshaped Sent Feature: ", sent_ft_reshaped.shape)
		sent_fts_list.append(sent_ft_reshaped)
		print("")
	sent_fts_arr = np.concatenate(sent_fts_list, axis=0)
	np.save('sent_fts_arr.npy', sent_fts_arr)
	return sent_fts_arr

def _get_corr_sentences(img_name):	
	img_desc = []
	for i in sent_file:
		if i.find(img_name) == -1:
			pass
		else:
			sent = ' '.join(i.split()[1:len(i.split())-1]) 
			img_desc.append(TaggedDocument(words=sent.split(), tags=[img_name]))
	return img_desc

def _get_corr_sentences_wrong(img_name):
	img_desc = []
	wrong_desc = []
	for i in sent_file:
		if i.find(img_name) == -1:
			img_desc.append(i)
		else:
			pass
	sent_list = random.sample(img_desc, 5)
	for i in sent_list:
		sent = ' '.join(i.split()[1:len(i.split())-1]) 
		wrong_desc.append(TaggedDocument(words=sent.split(), tags=[img_name]))
	return wrong_desc

def next_batch(data, size):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:size]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)
