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
        return(np.matrix([[np.random.normal() for i in range(layer1)] for j in range(layer2)]))
    
    # Backpropagation. As above.
    def back(self,inputs,weights):
        return(list(map(lambda x: drelu(x),weights*inputs)))

    # One hot encoder. Expects array of integer labels, and the total number of labels.
    def onehot(self,labels,nlabels=3):
        return(list(map(lambda x: [1 if i == x else 0 for i in range(nlabels)], labels)))
    
    # Cost function. Takes in arrays of labels and corresponding output values.
    def cost(self,right,wrong):
        return(np.sum((-right * np.log(wrong)) - ((1-right) * np.log(1-wrong))))
    
    #Adds +1 bias to layer.
    def bias(self,layer):
        return(np.append(layer,np.ones(len(layer))))
#        return(np.append([layer],[np.ones(len(layer))],axis=0))
    
    # Forward step. Mapping relu to weights*inputs vector. Weights is a matrix, inputs a column vector.
    # NEEDS BIAS.
    def output(self,inputs,weights):
        if self.act == "elu":
            return(list(map(lambda x: elu(x) if x != 1 else x, weights*inputs)))
        elif self.act == "relu":
            return(list(map(lambda x: relu(x) if x != 1 else x, weights*inputs)))
        elif self.act == "sigmoid":
            return(list(map(lambda x: sigmoid(x) if x != 1 else x, weights*inputs)))

    # Forward pass.
    def forward(self,datapoint):
        return(reduce(lambda x,y: output(x,y), [datapoint,*weights]))

    # Calculates outputs. Decided to forego 
    def outs(self,datapoint):
        out = [output(datapoint,weights[0])]
        for i in range(1,self.layers):
            out.append(output(out[-1],weights[i]))
        return(out)

    def outs(self,datapoint)
        return([output(datapoint,weights[0]),
                list(map(lambda x,y: output(x,y), 
                         [output(datapoint,weights[0]),*weights[1:]]))])
    
            
        return([output(self.data[0],self.weights[0]),
                *list(map(lambda x: output(self.data[x],self.weights[x]), range(1,self.layers)))])

    # Converting input to column vector form.
    coldata = [[i] for i in self.data]
    
    # Generating random weights. The "*2" is to account for bias nodes.
    weights = [initweights(data.T,range(nnodes)),
               initweights(range(nnodes*2),range(nnodes)) for i in range(nlayers-1),
               initweights(range(nnodes),len(set(target)))]
    
    for i in range(self.cycles):
        for j in range(self.nodes):
            
