import tensorflow as tf
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os, random
import pandas as pd
import re
import seaborn as sns
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import random
from tensorflow.contrib.layers.python.layers import fully_connected
import network, data_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_EPOCH_ = 1000
BATCH_SIZE = 40
SAMPLE_RATIO = 2
LEARNING_RATE = 0.001
getBatch = True

input_text = tf.placeholder(tf.string, shape=(None))
input_image = tf.placeholder(tf.float32, shape=(None))
labels_phl = tf.placeholder(tf.bool, shape=([BATCH_SIZE*SAMPLE_RATIO,BATCH_SIZE]))

module_img = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=False)
headers = ["img","caption"]
df = pd.read_csv('/home/super/datasets/flicker30/results_20130124_orig.token',names = headers,delimiter="\t")
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
embed = hub.Module(module_url, trainable=True)

session = tf.InteractiveSession()
global_step = tf.Variable(0, trainable=False)
images = tf.image.convert_image_dtype(input_image, tf.float32)
img_emb = module_img(images)
sent_emb = embed(input_text)
i_emb, s_emb = network.get_network(img_emb, sent_emb)
loss = network.embedding_loss(i_emb, s_emb, labels_phl)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss, global_step=global_step)
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
saver = tf.train.Saver()

for epoch in range(NUM_EPOCH_): # set total epochs
    print("Epoch: ",epoch)
    for epoch2 in range(2): # epochs per mini batch
        Image_x = []
        Image_x2 = []
        Text_x = []
        Label_y = []
        j = 0
        if(getBatch):
            ran = random.randint(0,len(df)-200)
            mini_batch = data_utils.get_batch(ran,ran+200, df)
        for d in mini_batch:
            Label_y_=[]
            for i in range(80):
                Label_y_.append(0)
            Image_x.append(d.img_feat)
            r = random.randint(0,4)
            Text_x.append(d.sentences[r])
            Label_y_[j] = 1
            r2 = random.randint(0,4)
            Text_x.append(d.sentences[r2])
            Label_y_[j+1] = 1
            Label_y.append(Label_y_)
            j += 2
        Image_x = np.asarray(Image_x)
        Label_y = np.asarray(Label_y)
        Label_y = Label_y.T
        for e in range (5):
            feed_dict = {
	            input_text: Text_x,
	            input_image: Image_x,
	            labels_phl: Label_y
            }
            _, loss_sess = session.run([train_step, loss], feed_dict=feed_dict)
            print("Loss: {0}".format(loss_sess))
print("Saving Model!")
saver.save(session, './model',global_step=global_step)
