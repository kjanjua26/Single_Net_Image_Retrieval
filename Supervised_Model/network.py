import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import fully_connected
import numpy as np 

def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def pdist(x1, x2):
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def get_network(img_emb, sent_emb):
	im_fc1 = add_fc(img_emb, 2048, True, 'im_embed_1')
	im_fc2 = fully_connected(im_fc1, 512, activation_fn=None, scope = 'im_embed_2')
	i_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)
	sent_fc1 = add_fc(sent_emb, 2048, True,'sent_embed_1')
	sent_fc2 = fully_connected(sent_fc1, 512, activation_fn=None, scope = 'sent_embed_2')
	s_embed = tf.nn.l2_normalize(sent_fc2, 1, epsilon=1e-10)
	return i_embed, s_embed

def embedding_loss(im_embeds, sent_embeds, im_labels):
    sent_im_ratio = 2 # Sample_size 
    num_img = 40 # Batch_Size
    num_sent = num_img * sent_im_ratio
    im_loss_factor = 1.5
    margin = 0.05
    sent_only_loss_factor=0.05
    sent_im_dist = pdist(sent_embeds, im_embeds)
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, im_labels), [num_sent, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~im_labels), [num_sent, -1])
    im_loss = tf.clip_by_value(margin+ pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=4)[0])
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)), [num_img, -1])
    neg_pair_dist = tf.reshape(tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
    sent_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=4)[0])
    sent_sent_dist = pdist(sent_embeds, sent_embeds)
    sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [1, sent_im_ratio]), [num_sent, num_sent])
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
    pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
    sent_only_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=4)[0])

    loss = im_loss * im_loss_factor + sent_loss + sent_only_loss * sent_only_loss_factor
    
    return loss
