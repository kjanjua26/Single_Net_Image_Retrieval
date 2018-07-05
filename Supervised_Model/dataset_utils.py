from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np 

class Data:
    def __init__(self, img_name, img_feat,sentences):
        self.img_name = img_name
        self.img_feat = img_feat
        self.sentences = sentences

def get_batch(start,end, df):
    data = [] 
    i =0 
    text_X = []
    label_X = []
    for img,cap in zip(df.img[start:end],df.caption[start:end]):
        img_split=img.split('#')
        img_path ='/home/super/datasets/flicker30/images/train/flickr30k_images/'+ img_split[0]
        img2 = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img2)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = x.reshape(299,299,3)
        text_X.append(cap)
        i += 1
        if(i % 5 == 0):
            data.append(Data(img_split[0],x,text_X))
            i =0 
            text_X = []
            label_X = []
    return data
