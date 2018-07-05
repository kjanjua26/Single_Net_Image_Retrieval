from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pandas as pd 
import os, re
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np 
from scipy import spatial
import network
from tensorflow.contrib.layers.python.layers import fully_connected
from scipy.spatial import distance

os.environ["CUDA_VISIBLE_DEVICES"] = ""

module_img = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=False)
headers = ["img","caption"]
df = pd.read_csv('/home/super/datasets/flicker30/results_20130124_orig.token', names=headers, delimiter="\t")
print("Image: ", df.img[1])
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
embed = hub.Module(module_url, trainable=True)
img_split = df.img[0].split('#')
img_path ='/home/super/datasets/flicker30/images/train/flickr30k_images/'+ img_split[0]

input_text = tf.placeholder(tf.string, shape=(None))
input_image = tf.placeholder(tf.float32, shape=(None))

loaded_img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(loaded_img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = x.reshape(299,299,3)
Image_x2 = []
Image_x2.append(x)
Image_x2 = np.asarray(Image_x2)
Text_x2 = []
Text_x2.append(df.caption[0])
print("Caption: ", df.caption[0])
images = tf.image.convert_image_dtype(input_image1, tf.float32)
img_emb = module_img(images)
s_emb = embed(input_text)
i_embed, s_embed = network.get_network(img_emb, s_emb)

sess = tf.Session()
sess.run(tf.tables_initializer())
saver = tf.train.Saver(save_relative_paths=True)
saver = tf.train.import_meta_graph('model-5000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

sen_embed, img_embed1 = sess.run([s_embed,i_embed], feed_dict={input_text:Text_x2, input_image1:Image_x2})
print("Distance between First Caption and First Image: ", distance.euclidean(sen_embed,img_embed1))
print("Cosine Distance: ", spatial.distance.cosine(sen_embed,img_embed1))
