import tensorflow as tf 

correct_X = tf.placeholder(tf.float32, [None, 2])
wrong_X = tf.placeholder(tf.float32, [None, 2])

def pairwise_loss(x1, x2):
    l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(x1, x2)), reduction_indices=1))
    margin = tf.constant(1.)     
    match_loss = tf.square(l2diff, 'match_loss')
    mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)), 'mismatch_term')
    loss = tf.add(1., tf.multiply(1., mismatch_loss), 'loss_add') # margin
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

def network(corr_X, neg_X):
	# for corr_X
	corr_fconn_1 = tf.contrib.layers.fully_connected(corr_X, 2048, activation_fn=None, scope='corr_fconn_1')
	corr_fconn_1_batch_norm = tf.layers.batch_normalization(corr_fconn_1, momentum=0.1, epsilon=1e-5, training=True, name='corr_fconn_1_batch_norm')
	corr_fconn_1_relu = tf.nn.relu(corr_fconn_1_batch_norm, name='corr_fconn_1_relu')
	corr_fconn_1_dropout = tf.layers.dropout(corr_fconn_1_relu, seed=0, training=True, name='corr_fconn_1_dropout')
	corr_fconn_2 = tf.contrib.layers.fully_connected(corr_fconn_1_dropout, 512, activation_fn=None, scope='corr_fconn_2')
	corr_X_emb = tf.nn.l2_normalize(corr_fconn_2, 1, epsilon=1e-10)

	# for neg_X
	neg_fconn_1 = tf.contrib.layers.fully_connected(neg_X, 2048, activation_fn=None, name='neg_fconn_1')
	neg_fconn_1_batch_norm = tf.layers.batch_normalization(neg_fconn_1, momentum=0.1, epsilon=1e-5, training=True, name='neg_fconn_1_batch_norm')
	neg_fconn_1_relu = tf.nn.relu(neg_fconn_1_batch_norm, name='neg_fconn_1_relu')
	neg_fconn_1_dropout = tf.layers.dropout(neg_fconn_1_relu, seed=0, training=True, name='neg_fconn_1_dropout')
	neg_fconn_2 = tf.contrib.layers.fully_connected(neg_fconn_1_dropout, 512, activation_fn=None, scope='neg_fconn_2')
	neg_X_emb = tf.nn.l2_normalize(neg_fconn_2, 1, epsilon=1e-10)

	return corr_X_emb, neg_X_emb

def compute_loss(corr_X, neg_X):
	corr_X_emb, neg_X_emb = network(corr_X, neg_X)
	loss = pairwise_loss(corr_X_emb, neg_X_emb)
	return loss
