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


def _get_training_data_corr():
	correct_list = []
	count = 0
	for i in glob.glob(img_path+"/*.jpg"):
		count += 1
		i_ed = i.split("/", 8)
		name = i_ed[8].replace(".jpg","")
		img_name = i
		img_ft = extract_features.get_img_features_vgg16(img_name)
		print("For Count: {0}, {1}".format(count,img_name))
		sent_list = _get_corr_sentences(name)
		sent_fts = extract_features.get_doc2v_model(sent_list)
		print("Sentence Feature: ", type(sent_fts), sent_fts.shape)
		print("Image Features: ", type(img_ft), img_ft.shape)
		img_ft_reshaped = np.reshape(img_ft, (-1, 2))
		sent_ft_reshaped = np.reshape(sent_fts, (-1,2))
		print("Reshaped Sent Feature: ", sent_ft_reshaped.shape)
		print("Reshaped Img Feature: ", img_ft_reshaped.shape)
		corr_fts = np.concatenate((img_ft_reshaped, sent_ft_reshaped), axis=0)
		print("Corr_Fts Shape: ", corr_fts.shape)
		correct_list.append(corr_fts)
		print("")
	correct_list_arr = np.concatenate(correct_list, axis=0)
	print("Shape of stacked: ", correct_list_arr.shape)
	np.save('corr_fts.npy', correct_list_arr)
	return correct_list_arr # returned feature array. 

def _get_training_data_wrong():
	wrong_list = []
	count = 0
	for i in glob.glob(img_path+"/*.jpg"):
		count += 1
		i_ed = i.split("/", 8)
		name = i_ed[8].replace(".jpg","")
		img_name = i
		img_ft = extract_features.get_img_features_vgg16(img_name)
		print("For Count: {0}, {1}".format(count,img_name))
		sent_list = _get_corr_sentences_wrong(name)
		sent_fts = extract_features.get_doc2v_model(sent_list)
		print("Sentence Feature: ", type(sent_fts), sent_fts.shape)
		print("Image Features: ", type(img_ft), img_ft.shape)
		img_ft_reshaped = np.reshape(img_ft, (-1, 2))
		sent_ft_reshaped = np.reshape(sent_fts, (-1,2))
		print("Reshaped Sent Feature: ", sent_ft_reshaped.shape)
		print("Reshaped Img Feature: ", img_ft_reshaped.shape)
		wrong_fts = np.concatenate((img_ft_reshaped, sent_ft_reshaped), axis=0)
		print("Wrong_Fts Shape: ", wrong_fts.shape)
		wrong_list.append(wrong_fts)
		print("")
	wrong_list_arr = np.concatenate(wrong_list, axis=0)
	print("Shape of stacked: ", wrong_list_arr.shape)
	np.save('wrong_fts.npy', wrong_list_arr)
	return wrong_list_arr # returned feature array. 


def next_batch(data, size):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:size]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)
