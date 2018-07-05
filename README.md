# Single_Net_Image_Retrieval
An End-to-End network for Multi Modal Deep Learning. 

## Unsupervised Approach
To train the unsupervised approach, run the file `python3 singleNet.py`. <br>
The model is label independent and uses the pairwise loss function to maximize the distance between negative pairs from the positive ones. 

## Supervised Approach
To train the supervised appraoch, run the file `Supervised_Model/train.py`.<br>
The models uses the label in form of one-hot vectors to make sure that the images and positive descriptions are nearer than images and their negative counterparts.
