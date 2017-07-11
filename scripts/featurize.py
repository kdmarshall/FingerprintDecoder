import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import MACCSkeys

from utils.utils import (ohe_label,
						 PAD_ID,
						 remove_salts)

max_char_len = 100
min_char_len = 20

data_dir = 'data/compounds'
out_file = 'data/features.npz'
sdf_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.sdf')]

fps = []
encoded_smiles = []
for sdf in sdf_list:
	print "Processing file %s"%sdf
	suppl = Chem.SDMolSupplier(sdf)
	for mol in tqdm(suppl):
		try:
			smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
		except:
			continue
		smiles = remove_salts(smiles)
		smiles_length = len(smiles)
		if smiles_length < min_char_len or smiles_length > max_char_len:
			continue
		fp = MACCSkeys.GenMACCSKeys(mol)
		fp_list = [int(bit) for bit in list(fp.ToBitString())[1:]]
		fps.append(fp_list)
		ohe = ohe_label(smiles)
		encoded_smiles.append(ohe)

samples = np.array(fps)
print "Samples shape {}".format(samples.shape)
labels = np.array(encoded_smiles)
print "Labels shape {}".format(labels.shape)

np.savez(out_file, samples=samples, labels=labels)

