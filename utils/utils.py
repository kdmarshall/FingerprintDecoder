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
		encoded = [OUTPUT_CHARS.index(c) for c in label_list]
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
	encoded_label = encode_label(label, pad=100)
	ohe = np.eye(VOCAB_SIZE)[encoded_label]
	return ohe

def decode_label(encoded_label):
	char_indicies = np.argmax(encoded_label, axis=1)
	return char_indicies

def decode_ohe(encoded_label):
	char_indicies = decode_label(encoded_label)
	raw_str = ''.join([OUTPUT_CHARS[idx] for idx in char_indicies])
	return raw_str

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
