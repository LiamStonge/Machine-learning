# Boltzmann Machine
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#We need to import the datase
movies = pd.read_csv('ml-lm/movies.dat', sep = '::', header = None, engine = 'python', enconding = 'latin-l')
users = pd.read_csv('ml-lm/users.dat', sep = '::', header = None, engine = 'python', enconding = 'latin-l')
ratings = pd.read_csv('ml-lm/ratings.dat', sep = '::', header = None, engine = 'python', enconding = 'latin-l')

#Next step is to create the training and test set for our model
training_set = pd.read_csv('ml-100k/ul.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/ul.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Now we need to get the number of users and movies so that we can work with any dataset
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_users = int(max(max(training_set[:,1]), max(test_set[:,1])))


#This is a simple method to convert the training and test set into a list of list for pytorch to work with
def convert(data):
	new_data = []
	for id_users in range(1, nb_users+1):
		id_movies = data[:,1][[data[:,0] == id_users]
		id_ratings = data[:, 2][data[:,0] == id_users]
		ratings = np.zeros(nb_movies)
		ratings[id_movies - 1] = id_ratings
		new_data.append(list(ratings))
	return new_data

#using the method to convert the training and test set into list of list
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into PyTorch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Making the ratings into 1 or 0 (liked, not liked)
# First change the 0s(not rated) into -1 
training_set[training_set == 0] = -1
# Convert all the ratings of the movies that the user didn't like into 0
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
# Convert all the ratings of the movies that the user didn't like into 0
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

class RBM():
	def __init__(self, nh, nv):
		self.w = torch.randn(nh, nv)
		self.a = torch.randn(1, nh)
		self.b = torch.randn(1, nv)
	def sample_h(self, x):
		wx = torch.mm(x, self.w.t())
		activation = wx + self.a.expand_as(wx)
		p_h_given_v = torch.sigmoid(activation)
		return p_h_given_v, torch.bernoulli(p_h_given_v)
	def sample_v(self, y):
		wy = torch.mm(y, self.w)
		activation = wy + self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		return p_v_given_h, torch.bernoulli(p_v_given_h)
	def train(self, v0, vk, ph0, phk):
		self.w += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
		self.b += torch.sum((v0-vk), 0)
		self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epochs in range(1, nb_epochs + 1):
	train_loss = 0
	s = 0.
	for id_user in range(0, nb_users - batch_size, batch_size):
		vk = training_set[id_user:id_user+batch_size]
	    v0 = training_set[id_user:id_user+batch_size]
		pho,_ = rbm.sample_h(v0)
		for k in range(10):
			_,hk = rbm.sample_h(vk)
			_,vk = rbm.sample_v(hk)
			vk[v0<0] = v0[v0<0]
		phk,_ = rbm.sample_h(vk)
		rbm.train(v0, vk, ph0, phk)
		train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
		s += 1.
	print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
	

test_loss = 0
s = 0.
for id_user in range(nb_users):
	v = training_set[id_user:id_user+1]
	vt = test_set[id_user:id_user+1]
	if len(vt[vt>=0]) > 0:	
		_,h = rbm.sample_h(v)
		_,v = rbm.sample_v(h)
		test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
		s += 1.
print('test loss: '+str(test_loss/s))










