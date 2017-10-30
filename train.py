import tensorflow as tf
import numpy as np
import logging
from utils import (DataSet,
				   decode_ohe,
				   clean_prediction)
from models import EncoderModel as Model

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset npz.')
tf.app.flags.DEFINE_integer("batch_size", 16, 'Size of train and valid batches.')
tf.app.flags.DEFINE_string("log", None, 'File to write log to.')
FLAGS = tf.app.flags.FLAGS
# log helper
def log_setup(log_flag):
    if log_flag:
        logging.basicConfig(filename=log_flag, level=logging.INFO, format='%(message)s')
        return logging.info
    else:
        return print
logger = log_setup(FLAGS.log)

def main(*args):
	# Hyper parameters
	learning_rate = 0.001
	training_steps = 1000000
	valid_step = 50
	cell_size = 256
	hidden_layers = (256, 256, 256, cell_size*2)
	num_rnn_layers = 3

	dataset = DataSet(FLAGS.dataset)
	train_model = Model(dataset.samples_shape[1],
				  dataset.labels_shape[1],
				  dataset.labels_shape[2],
				  cell_size,
				  num_rnn_layers,
				  hidden_layers,
				  lr=learning_rate,
				  cell_type='lstm')

	valid_model = Model(dataset.samples_shape[1],
				  dataset.labels_shape[1],
				  dataset.labels_shape[2],
				  cell_size,
				  num_rnn_layers,
				  hidden_layers,
				  training=False,
				  lr=learning_rate,
				  cell_type='lstm')

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		loss = []
		for step in range(training_steps):
			train_samples, train_labels, train_weights = dataset.get_batch(FLAGS.batch_size, 'train')
			train_labels_T = np.transpose(train_labels, (1, 0, 2))
			_loss, prediction = train_model.step(train_samples, train_labels_T, train_weights, sess)
			loss.append(_loss)
			if (step % valid_step) == 0:
				logger("Average training loss: %s" % np.mean(loss))
				loss = []
				valid_samples, valid_labels, valid_weights = dataset.get_batch(FLAGS.batch_size, 'valid')
				valid_labels_T = np.transpose(valid_labels, (1, 0, 2))
				_, v_prediction = valid_model.step(valid_samples, valid_labels_T, valid_weights, sess, valid=True)
				logger("Valid @ step %s" % (step,))
				for p in v_prediction[:5]:
					pred = decode_ohe(p)
					cleaned_pred = clean_prediction(pred)
					logger(cleaned_pred)

				# correct_prediction = tf.equal(tf.argmax(), tf.argmax(valid_labels))
				# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				# raw_accuracy = sess.run(accuracy, feed_dict={x: v_prediction, y_: mnist.test.labels})
				# print(model.accuracy(v_prediction, valid_labels))

if __name__ == "__main__":
    tf.app.run()
