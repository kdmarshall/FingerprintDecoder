import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import (LSTMCell,
									GRUCell,
									MultiRNNCell,
									LSTMStateTuple)
import tensorflow.contrib.seq2seq as seq2seq

from utils import (EOS_ID,
				   PAD_ID,
				   GO_ID)

class EncoderModel(object):
	"""Core training model"""
	def __init__(self, input_shape, max_seq_len, vocab_size, cell_size, num_layers, hidden_layers, training=True, lr=5e-5, cell_type='lstm'):

		self.vocab_size = vocab_size
		self.cell_size = cell_size
		self.training = training

		# Placeholders
		self.input_node = tf.placeholder(tf.float32, [None, input_shape])
		self.label_node = tf.placeholder(tf.float32, [max_seq_len, None, vocab_size])
		self.label_weights = tf.placeholder(tf.float32, [None, max_seq_len])

		layers = []
		with slim.arg_scope([slim.fully_connected]):
			for layer_index, layer_size in enumerate(hidden_layers):
				_scope = "layer%s" % layer_index
				if layer_index == 0:
					layers.append(slim.fully_connected(self.input_node, layer_size, reuse=True, scope=_scope))
				else:
					layers.append(slim.fully_connected(layers[-1], layer_size, reuse=True, scope=_scope))
		out_layer = layers[-1]
		
		if cell_type == 'gru':
			cell_class = GRUCell
		elif cell_type == 'lstm':
			cell_class = LSTMCell
		else:
			raise ValueError("Cell type '%s' not valid"%cell_type)

		if num_layers > 1:
			self.decoder_cell = MultiRNNCell([cell_class(cell_size, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
		else:
			self.decoder_cell = cell_class(cell_size, state_is_tuple=True)

		initial_input, initial_state = out_layer[:,:cell_size], out_layer[:,cell_size:]
		if num_layers > 1:
			state_vector = [LSTMStateTuple(c=initial_state, h=initial_input) for _ in range(num_layers)]
		else:
			state_vector = LSTMStateTuple(
	    		c=initial_state,
	    		h=initial_input
			)

		if self.training:
			with tf.variable_scope("rnn") as varscope:
				logits_list = []
				for i in range(max_seq_len):
					if i > 0:
						tf.get_variable_scope().reuse_variables()
					time_input = self.label_node[i, :, :]
					output, state_vector = self.decoder_cell(time_input, state_vector)
					output_logits = self.project_to_chars(output)
					logits_list.append(output_logits)
		else:
			with tf.variable_scope("rnn") as varscope:
				logits_list = []
				for i in range(max_seq_len):
					if i > 0:
						time_input = tf.one_hot(tf.argmax(output_logits, axis=-1), self.vocab_size)
					else:
						time_input = self.label_node[0, :, :]
					tf.get_variable_scope().reuse_variables()
					output, state_vector = self.decoder_cell(time_input, state_vector)
					output_logits = self.project_to_chars(output)
					logits_list.append(output_logits)


		logits_tensor = tf.stack(logits_list)
		logits_tensor_T = tf.transpose(logits_tensor, [1,0,2])
		self.softmax_logits = tf.nn.softmax(logits_tensor_T)
		if self.training:
			label_node_T = tf.transpose(self.label_node, [1,0,2])
			dense_labels = tf.argmax(label_node_T, axis=2)
			self.loss = seq2seq.sequence_loss(logits_tensor_T,
											  dense_labels,
											  self.label_weights)
			self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)


	def project_to_chars(self, input):
		weights = tf.get_variable('project_chars_weights',
								  shape=[self.cell_size, self.vocab_size],
								  initializer=tf.random_normal_initializer())
		bias = tf.get_variable('project_chars_bias',
								shape=[self.vocab_size],
								initializer=tf.random_normal_initializer())
		return tf.add(tf.matmul(input, weights), bias)


	def step(self, input_batch, label_batch, length_batch, sess, valid=False):
		if valid:
			pred = sess.run(self.softmax_logits,
							 feed_dict={
							  self.input_node: input_batch,
							  self.label_node: label_batch,
	                          self.label_weights: length_batch
							 })
			return None, pred
		else:
			l, pred, _ = sess.run([self.loss, self.softmax_logits, self.optimizer],
								   feed_dict={
								    self.input_node: input_batch,
								    self.label_node: label_batch,
		                          	self.label_weights: length_batch
								   })
		return l, pred


	def accuracy(self, predicted_batch, true_batch):
		# correct_prediction = tf.equal(tf.argmax(), tf.argmax(valid_labels))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# raw_accuracy = sess.run(accuracy, feed_dict={x: v_prediction, y_: mnist.test.labels})
		correct_prediction = np.equal(np.argmax(predicted_batch, np.argmax(true_batch)))
		accuracy = np.mean(correct_prediction.astype(np.float32))
		return accuracy

