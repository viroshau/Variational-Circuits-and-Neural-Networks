from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf

nodes = 10
renyuigraph = False
if renyuigraph:
    betabound = (0,np.pi/2)
    gammabound = (0,2*np.pi)
    graph = nx.fast_gnp_random_graph(nodes,0.5,seed = 123)
else:
    betabound = (0,np.pi/2)
    gammabound = (0,np.pi)
    graph = nx.generators.random_regular_graph(3,nodes,seed = 123)

nqubits = len(graph.nodes())
print(nqubits)
#classicalresult = bestClassicalHeuristicResult(graph)
cost_h, mixer_h = qml.qaoa.maxcut(graph)

dev = qml.device('default.qubit', wires=nqubits,shots = 1)
@qml.qnode(dev)
def circuit(inputs,weights):
    #qml.AngleEmbedding(inputs,wires = range(nqubits))
    for i in range(nqubits):
        qml.Hadamard(wires = i)
    
    p = len(weights)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-weights[:p][i], wires = k)
            qml.CNOT(wires = [j,k])

        #for j,k,w in edges:
            #qml.CNOT(wires = [int(j.item()),int(k.item())])
            #qml.RZ(-x[:p][i], wires = int(k.item()))
            #qml.CNOT(wires = [int(j.item()),int(k.item())])

        for j in range(nqubits):
            qml.RX(2*weights[p:][i],wires = j)

    return qml.sample()

#print(circuit(1,np.array([1,2])))

layers = 4
weight_shapes = {'weights':(layers,)} #Details of the shape of each trainable parameter in the qnode

qlayer = qml.qnn.TorchLayer(circuit,weight_shapes) #Turning a quantum node into a torch layer. This can be incorporated into any torch workflow.

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit,weight_shapes)
        self.clayer = torch.nn.Linear(nqubits,nqubits) #n_qubits-sized input, n_qubits-sized output. Layers are fully connected
        self.activationfunction = torch.nn.Tanh()

    def forward(self,x):
        x = self.qlayer(x)
        x = x.type(torch.float) #Do something about this
        x = self.clayer(x)
        return self.activationfunction(x)

model = HybridModel()
print(model(torch.tensor([[1,2,3,4,5,6,7,8,9,10]])))

print(list(range(8)))
