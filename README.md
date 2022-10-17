# SmallMolecules_MoIGAN_HPT
Generating hypertension small molecules using MoIGAN and Hypertension dataset from ChEMBL.

Reference: https://colab.research.google.com/github/deepchem/deepchem/blob/master/examples/tutorials/Generating_molecules_with_MolGAN.ipynb#scrollTo=VQ9mMmseaVH_

"""
The following model was a product of the paper "MolGAN: An implicit generative model for small molecular graphs" by Cao and Kipf.
GAN = Generative Adversarial Network.

The architecture consits of 3 main sections: a generator, a dicriminator, and a reward network.

The generator takes a sample (z) from a standard normal distribution to generate an a graph using a MLP (this limits the network to a fixed maximum size) to generate the graph at once. Sepcifically a dense adjacency tensor A (bond types) and an annotation matrix X (atom types) are produced. Since these are probabilities, a discrete, sparse x and a are generated through categorical sampling.

The discriminator and reward network have the same architectures and recieve graphs as inputs. A Relational-GCN and MLPs are used to produce the singular output.
"""
