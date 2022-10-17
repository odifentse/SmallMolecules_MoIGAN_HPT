# Generating small molecules using MoIGAN

# Reference: https://colab.research.google.com/github/deepchem/deepchem/blob/master/examples/tutorials/Generating_molecules_with_MolGAN.ipynb#scrollTo=VQ9mMmseaVH_

"""
The following model was a product of the paper "MolGAN: An implicit generative model for small molecular graphs" by Cao and Kipf.
GAN = Generative Adversarial Network.

The architecture consits of 3 main sections: a generator, a dicriminator, and a reward network.

The generator takes a sample (z) from a standard normal distribution to generate an a graph using a MLP (this limits the network to a fixed maximum size) to generate the graph at once. Sepcifically a dense adjacency tensor A (bond types) and an annotation matrix X (atom types) are produced. Since these are probabilities, a discrete, sparse x and a are generated through categorical sampling.

The discriminator and reward network have the same architectures and recieve graphs as inputs. A Relational-GCN and MLPs are used to produce the singular output.
"""

# STEP 1: IMPORT LIBRARIES

import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict

import deepchem as dc
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit import Chem
# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

# STEP 2: ACCESS THE MOLECULE DATASET

# Download TOX21 dataset from MolNet
# tasks, datasets, transformers = dc.molnet.load_tox21(reload=False)
df = pd.read_csv('/Users/odilehasa/PyCharmProjects_Aug2022/GeneratingSmallMolecules_MoIGAN/HPT_Smiles.csv')
df   # Display the total dataset
df.dtypes

# STEP 3: SPECIFY THE MAX NUMBER OF ATOMS TO ENCODE FOR THE FEATURIZER AND MOIGAN NETWORK

# The higher the number of atoms the more data you will have in the dataset, which will make the model more complex (as the input dimensions become higher).

num_atoms = 12

data = df
data

# STEP 4: INITIALIZE THE FEATURIZER WITH THE MAXIMUM NUMBER OF ATOMS PER MOLECULE

# Create the featurizer. The more atom_labels you add, the more data you will have and the model gets more complex/unstable.
feat = dc.feat.MolGanFeaturizer(max_atom_count= num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14]) #15, 16, 17, 19, 20, 24, 29, 35, 53, 80])

# STEP 5: EXTRACT THE SMILES FROM THE DATAFRAME AS A LIST OF STRINGS
smiles = data['Smiles'].values
smiles

#
# no_blanks = [x for x in smiles if np.isnan(x) == False]
# no_blanks

# STEP 6: FILTER OUT MOLECULES WITH TOO MANY ATOMS

# This will enable us to reduce the model complexity.
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]
filtered_smiles # produces a condensed (shorter) list of the original set of smiles from data/df (now 2081)

# STEP 7: FEATURIZE THE MOLECULES

# Determine the set of features for a molecule.
features = feat.featurize(filtered_smiles)

# STEP 8: REMOVE ALL THE INVALID MOLECULES

indices = [i for i, data in enumerate(features) if type(data) is GraphMatrix]
print(indices)
features = [features[i] for i in indices]  # The result is an even more shortened list of molecules (now 1371).

# STEP 9: CREATE AN INSTANCE (INSTANTIATE) THE MOIGAN MODEL AND SET THE LEARNING RATE & MAX NUMBER OF ATOMS AS THE SIZE OF THE VERTICES.

gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])

# STEP 10: DEFINE THE ITERBATCHES

# Define the iterbatches function because gan_fit function needs iterable batches
def iterbatches(epochs):
    """
    :param epochs:
    :return:
    """
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}

# STEP 11: TRAIN THE GAN MODEL

# Train the model using the fit_gan function and generate new molecules with the predict_gan_generator function.
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

generated_data = gan.predict_gan_generator(1000) # Projected outcome of 1000 generated molecules


# STEP 12: CONVERT THE GENERATED GRAPHS TO RDKIT MOLECULES

nmols = feat.defeaturize(generated_data)
print("{} molecules generated".format(len(nmols))) # To print a statement of the number of molecules generated.


# STEP 13: REMOVE INVALID MOLECULES FROM THE LIST.

nmols = list(filter(lambda x: x is not None, nmols))  #lambda is a shortened if statement. Here we only want those molecules that don't have a blank value.


# STEP 14: PRINT OUT THE NUMBER OF VALID MOLECULES

# Training may be unstable and thus cause the model to produce a varying number of valid molecules.

print("{} valid molecules".format(len(nmols)))

# STEP 15: REMOVE DUPLICATE MOLECULES, FROM THE GENERATED MOLECULES.

nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print("{} unique valid molecules".format(len(nmols_viz)))


# STEP 16: PRINT OUT A LIST OF THE VALID AND UNIQUE MOLECULES

print(nmols_smiles_unique)
