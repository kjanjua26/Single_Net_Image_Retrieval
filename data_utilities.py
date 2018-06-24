import tensorflow as tf 
import numpy as np 
import pandas as pd 
import math, glob
import extract_features

sent_path = 'flicker-sample/sentences-flicker.txt' # a text file containing sentences
img_path = 'flicker-sample/images'
sent_file = open(sent_path, 'r').readlines()

def _get_sentences(img_name):	
	for i in sent_file:
		if i.find(img_name) == -1:
			pass
		else:
			return ' '.join(i.split()[1:len(i.split())-1]) 

for i in glob.glob(img_path+"/*.jpg"):
	sent_ft_lst = []
	i = i.split("/", 2)
	name = i[2].replace(".jpg","")
	img_name = "flicker-sample/images/" + i[2]
	img_ft = extract_features.get_img_features_vgg16(img_name)
	print "For {0}".format(img_name)
	sents_rt = _get_sentences(name)
	sent_fts = extract_features.get_doc2v_model(sents_rt) # doesn't work properly.
	print "Feature: ", type(sent_fts)
	sent_ft_lst.append(sent_fts)
	print "Features: ", type(img_ft)
	print ""
