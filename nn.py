# Neural network from scratch. Very much work in progress.

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

# Trying to keep functions as succint as possible.

class nn():
    def __init__(self,datain,targetin,nlayers=1,nnodes=10,alpha=1,act="elu",ncycles=10000):
        self.data = datain
        self.target = targetin
        self.layers = nlayers
        self.nodes = nnodes
        self.a = alpha
        self.act = act
        self.cycles = ncycles

    # ELU activation function (Clevert et al.: "FAST AND ACCURATE DEEP 
    # NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)").
    def elu(self,x):
        return(x if x > 0 else self.a * (np.exp(x)-1))
    
    # ELU derivative.
    def delu(self,x):
        return(x if x > 0 else elu(x) + self.a)
        
    # ReLU activation function.
    def relu(self,x):
        return(max(x,0))
    
    # ReLU derivative.
    def drelu(self,x):
        return(1 if x > 0 else 0)
    
    # Sigmoid activation function.
    def sigmoid(self,x):
        return(1/(1 + np.exp(-x)))
    
    # Sigmoid derivative.
    def dsigmoid(self,x):
        return(sigmoid(x) * (1-sigmoid(x)))

    # Random weights initializer.
    def initweights(self,layer1,layer2):
        return([[np.random.normal() for i in range(layer1)] for j in range(layer2)])
    
    # Forward pass. Mapping relu to weights*inputs vector. Weights is a matrix, inputs a column vector.
    def forward(self,inputs,weights):
        return(list(map(lambda x: relu(x),weights*inputs)))
    
    # Backpropagation. As above.
    def back(self,inputs,weights):
        return(list(map(lambda x: drelu(x),weights*inputs)))

    # One hot encoder. Expects array of integer labels, and the total number of labels.
    def onehot(self,labels,nlabels=3):
        return(list(map(lambda x: [1 if i == x else 0 for i in range(nlabels)], labels)))
    
    # Cost function. Takes in arrays of labels and corresponding output values.
    def cost(self,right,wrong):
        return(np.sum((-right * np.log(wrong)) - ((1-right) * np.log(1-wrong))))
        
    # Converting row vector to column form.
    def tocol(self, x):
        return([[i] for i in self.x])
    
    # Generating random weights.
    weights = [initweights(data,range(nnodes)),initweights(range(nnodes),range(nnodes)) for i in range(nlayers-1),
              initweights(range(nnodes),len(set(target)))]

