import numpy as np
import os
import sys
import random
import math

OUTPUT_CHARS = ['#', ')', '(', '+', '-', ',', '/', '.', '1', '0',
                    '3', '2', '5', '4', '7', '6', '9', '8', ':', '=', 'A',
                    '@', 'C', 'B', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                    'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', '[', 'Z',
                    ']', '\\',  'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                    'h', 'l', 'o', 'n', 's', 'r', 'u', 't',
                    '*', 'EOS_ID', 'GO_ID', '<>'
                    ]

VOCAB_SIZE = len(OUTPUT_CHARS)
UNK_ID = VOCAB_SIZE - 4
EOS_ID = VOCAB_SIZE - 3
GO_ID = VOCAB_SIZE - 2
PAD_ID = VOCAB_SIZE - 1

def encode_label(label, pad=None):
	label_list = list(label)
	try:
		encoded = [GO_ID]+[OUTPUT_CHARS.index(c) for c in label_list] + [EOS_ID]
		if pad:
			orig_len = len(encoded)
			pad_len = pad - orig_len
			if pad_len > 0:
				encoded = encoded + [PAD_ID for _ in range(pad_len)]
	except Exception as e:
		print(e)
		sys.exit(label)
	return encoded

def ohe_label(label):
	encoded_label = encode_label(label, pad=100+2)
	ohe = np.eye(VOCAB_SIZE)[encoded_label]
	return ohe

def decode_label(encoded_label):
	char_idx = np.argmax(encoded_label)
	return char_idx

def decode_ohe(encoded_label):
	char_indicies = [decode_label(char_ohe) for char_ohe in encoded_label]
	raw_str = ''.join([OUTPUT_CHARS[idx] for idx in char_indicies])
	return raw_str

def clean_prediction(prediction):
	pred = prediction.strip(OUTPUT_CHARS[GO_ID])
	eos_idx = pred.find('EOS_ID')
	if eos_idx == -1:
		return pred
	else:
		return pred[:eos_idx]

def remove_salts(smiles):
    """
    Removes salt component of the SMILES.
    Args:
        smiles (str): SMILES to remove salt from.
        
    Returns:
        str: SMILES with salt removed, if it existed.
    """
    
    if '.' in smiles:
        fragments = smiles.split('.')
        max_length,longest_frag = max([(len(frag),frag) for frag in fragments])
        return longest_frag
    else:
        return smiles

class DataSet(object):
	def __init__(self, file_path, valid_size=0.2):
		self.file_path = file_path
		self.npzfile = np.load(self.file_path)
		self.files = self.npzfile.files
		self.samples = self.npzfile['samples']
		self.labels = self.npzfile['labels']
		self.weights = self.npzfile['weights']
		assert self.samples.shape[0] == self.labels.shape[0], "Number of samples and labels are not equal"
		self.set_size = self.samples.shape[0]
		total_indices = list(range(self.set_size))
		random.shuffle(total_indices)
		valid_partition = math.floor(self.set_size*valid_size)
		train_indices = total_indices[:-valid_partition]
		valid_indicies = total_indices[-valid_partition:]

		self.train_samples = self.samples[train_indices]
		self.train_labels = self.labels[train_indices]
		self.train_weights = self.weights[train_indices]

		self.valid_samples = self.samples[valid_indicies]
		self.valid_labels = self.labels[valid_indicies]
		self.valid_weights = self.weights[valid_indicies]


	@property
	def samples_shape(self):
	    return self.samples.shape
	
	@property
	def labels_shape(self):
	    return self.labels.shape

	@property
	def weights_shape(self):
	    return self.weights.shape
	
	def get_batch(self, batch_size, batch_type):
		assert batch_type in ('train','valid'), "Unrecognized batch_type %s" % batch_type

		def batch_helper(samples, labels, weights):
			indices = np.random.randint(0, samples.shape[0], batch_size)
			samples_batch = samples[indices]
			labels_batch = labels[indices]
			weights_batch = weights[indices]
			return (samples_batch, labels_batch, weights_batch)

		if batch_type == 'train':
			samples_batch, labels_batch, weights_batch = batch_helper(self.train_samples, self.train_labels, self.train_weights)
		else:
			samples_batch, labels_batch, weights_batch = batch_helper(self.valid_samples, self.valid_labels, self.valid_weights)
		return (samples_batch, labels_batch, weights_batch)
