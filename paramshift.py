import pennylane as qml
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from graphCreation import *
from QuantumCircuits import *
# set the random seed
#np.random.seed(42)


# create a device to execute the circuit on
G = CreateRegularGraph(8,4,randomweights = False,seed = None)
clEnergy,left,right = BestClassicalHeuristicResult(G)
adjacecymatrix = CreateAdjacencyMatrix(G)
NUMSHOTS = 10
dev = qml.device("default.qubit.torch", wires=len(G.nodes),shots = NUMSHOTS)
devExact = qml.device('default.qubit.torch', wires = len(G.nodes))

shotCircuit = qml.QNode(QAOAreturnSamples2,dev,interface = 'torch') #Returns a set of measurement samples
costHamiltonianCircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best')
cost_h = createCostHamiltonian(G,adjacecymatrix)

@qml.qnode(dev,interface = 'torch')
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
    return qml.sample()

class OneLayerNN(nn.Module):
    def __init__(self,D_in,D_out):
        super(OneLayerNN,self).__init__()
        self.linear = torch.nn.Linear(D_in,D_out,bias = False)
        self.tanh = nn.Tanh() #[-1,1]  #Potentially look into using hardtanh instead
    def forward(self,x):
        x = self.linear(x)
        x = self.tanh(x) 
        return x
    
    def set_custom_weights(self,model,weight_matrix):
        #Set the weight_matrix into a custom matrix defined as an argument to this function
        for name,param in model.named_parameters():
            param.data = weight_matrix
    
    def return_weights(self):
        return self.linear.weight.data

model = OneLayerNN(len(G.nodes),len(G.nodes))
model.set_custom_weights(model,torch.eye(len(G.nodes)))
print(model.return_weights())

def parameter_shift_term(qnode,params,i):
    #Parameter shift method with respect to i-th parameter
    shifted = torch.clone(params)

    #Forward shift
    epsilon = 0.01
    shifted[i]+=epsilon
    samples = torch.reshape(2*qnode(shifted,G)-1,shape = (NUMSHOTS,len(G.nodes))).type(torch.float32)
    samples = torch.sign(model(samples))
    forward = EvaluateCutOnDataset(samples,adjacecymatrix)
    #print(model(torch.Tensor(forward)))

    #Backward shift
    shifted[i]-=epsilon
    samples = torch.reshape(2*qnode(shifted,G)-1,shape = (NUMSHOTS,len(G.nodes))).type(torch.float32) #Runs circuit, gives samples
    samples = torch.sign(model(samples))
    backward = EvaluateCutOnDataset(samples,adjacecymatrix)
    #print(model(torch.Tensor(backward)))
    return 0.5*(torch.mean(forward)-torch.mean(backward))

def parameter_shift(qnode,params):
    gradients = np.zeros(len(params))
    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode,params,i)
    return gradients

params = torch.Tensor(7*np.random.random(10)).type(torch.float64)

energies = torch.zeros(100)
init_params = params
for i in range(100):
    params = params - 0.01*parameter_shift(shotCircuit,params)
    samples = torch.reshape(2*shotCircuit(params,G)-1,shape = (NUMSHOTS,len(G.nodes))).type(torch.float32)
    samples = torch.sign(model(samples))
    energies[i] = torch.mean(EvaluateCutOnDataset(samples,adjacecymatrix))
    if i % 20 == 0:
        print(params)


plt.figure()
plt.plot(list(range(100)),energies.detach().numpy())
plt.show()
print('Init_params:',init_params)
#print(parameter_shift(circuit,params))
#grad_function = qml.grad(circuit)
#print(grad_function(params))