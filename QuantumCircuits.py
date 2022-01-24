from mimetypes import init
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from graphCreation import *
import networkx as nx
import matplotlib.pyplot as plt

def QAOAcircuit(gammas,betas,G):
    """Creates the QAOA circuit structure. This function does not return any observable, but is instead called on from other functions where a specific return is required. 

    Args:
        gammas ([torch.tensor]): [A 1D array containing all gamma_i = [gamma_1,...,gamma_p]]
        betas ([torch.tensor]): [A 1D array containing all beta_i = [gamma_1,...,gamma_p]]
        G ([nx.graph]): [The graph to compute the circuit from]
    """
    #Layer of hadamards
    for i in G.nodes:
        qml.Hadamard(wires = i)

    #For p QAOA layers, alternatively apply mixer and problem unitaries
    for p in range(len(gammas)):
        # Problem unitary
        for i,j,w in G.edges(data = True):
            qml.CNOT(wires = [i,j])
            qml.RZ(gammas[p]*w['weight'],wires = j) #minus or no minus?
            qml.CNOT(wires = [i,j])
        
        # Mixer unitary
        for i in G.nodes:
            qml.RX(betas[p],wires = i)

def QAOAreturnCostHamiltonian(gammas,betas,G,cost):
    QAOAcircuit(gammas,betas,G)
    return qml.expval(cost)

def QAOAreturnSamples(gammas,betas,G):
    """Generate computational basis state measurements instead using a shot-based device

    Args:
        gammas ([1D array]): [Array of length p with all the gammas]
        betas ([1D arry]): [Array of length p with all betas]
        G ([nx.graph]): [The graph of which the QAOA procedure is performed on]

    Returns:
        [2D array]: [Array of shape (shots,wires)]
    """
    QAOAcircuit(gammas,betas,G)
    return qml.sample(op = None,wires = range(len(G.nodes)))

def QAOAreturnProbs(gammas,betas,G):
    QAOAcircuit(gammas,betas,G)
    return qml.probs(wires = range(len(G.nodes)))

def QAOAreturnPauliZExpectation(gammas,betas,G):
    QAOAcircuit(gammas,betas,G)
    return [qml.expval(qml.PauliZ(wires = i)) for i in range(len(G.nodes))]

def listofBinaryToInt(binarylist):
    descimalnumber = 0
    for i in range(len(binarylist)):
        if binarylist[len(binarylist)-i-1] == 1:
            descimalnumber += 2**i
    return descimalnumber

def changingNNWeights(initial_weights,iterations,i,g):
    result_matrix = (1-g(i,iterations))*initial_weights + g(i,iterations)*torch.ones_like(initial_weights)
    return result_matrix

#Define graph
graphlist = []
for i in [0,1]:
    for j in [2,3]:
        graphlist.append((i,j,1))
G = CreateGraphFromList(graphlist)
G = CreateRegularGraph(8,3)
print(G.nodes)
#DrawGraph(G)
clEnergy,_,_ = BestClassicalHeuristicResult(G)
cost_h, mixer_h = qml.qaoa.maxcut(G)

#Define Devices to run the circuit
NUMSHOTS = 1000
devExact = qml.device('default.qubit.torch', wires = len(G.nodes)) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
devShot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = NUMSHOTS)

#Turn gammas and betas into trainable variables
gammas = torch.tensor([0.1,2,3,4,6,7],dtype = torch.float64)
gammas = torch.autograd.Variable(gammas,requires_grad = True)
betas = torch.tensor([0.1,2,3,4,9,2],dtype = torch.float64)
betas = torch.autograd.Variable(betas,requires_grad = True)

qcircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best') #Can change the diff.method during the creation of the Qnode

#Perform steps for optimization
iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
print(f'Initial Parameters: \nGamma = {gammas} \nBeta = {betas}')
for i in range(iterations):
    opt.zero_grad()
    #print(f'Gammas = {gammas}, Betas = {betas}')
    loss = qcircuit(gammas,betas,G = G,cost = cost_h) 
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 20 == 0:
        print(f'Current progress: {i}/{iterations}, Current approximation ratio: {-1*loss.item()/clEnergy}') 
print(f'Final Parameters: \n Gamma = {gammas} \n Beta = {betas}')

#Sample form the circuit to generate the strings that the NN should be trained with

samplingqcircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch',diff_method = 'best')

samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes))) #Returns the samples from the circuit of shape (NUMSHOTS,numqubits)

#Define the NN

class OneLayerNN(nn.Module):
    def __init__(self,D_in,D_out):
        super(OneLayerNN,self).__init__()
        self.linear = torch.nn.Linear(D_in,D_out,bias = False)
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = self.linear(x)
        x = self.tanh(x)
        return x
    
    def set_custom_weights(self,model,weight_matrix):
        #Set the weight_matrix into a custom matrix defined as an argument to this function
        for name,param in model.named_parameters():
            param.data = weight_matrix

#Train the Neural Network
model = OneLayerNN(len(G.nodes),len(G.nodes))

tot_epoch = 1
iterationsNN = 40
opt = torch.optim.Adam(model.parameters(),lr = 0.3)

for epoch in range(tot_epoch):
    #For each epoch, create a new set of samples from the VQC
    samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes)))
    samples = (2*samples-1)
    samples = samples.type(torch.float32)

    for i in range(iterationsNN):
        #For each of the samples, perform a model prediction
        predictions = model(samples)
        
        #Calculate the loss on the dataset
        loss = EvaluateCutOnDataset(predictions,G) 

        #Perform backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        #Print status of the simulation
        if i % 10 == 0:
            print(f'Epoch: {epoch}/{tot_epoch}. Current progress: {i}/{iterationsNN}, Current approximation ratio: {-1*loss.item()/clEnergy}') 

#Train variational circuit where the outputs from the circuit is passed through a neural network

def customcost(gammas,betas,G,qcircuit,neuralNet):
    x = (qcircuit(gammas,betas,G)).float() #one shot to the circuit
    x = neuralNet(x) #pass it through the neural network
    print(x)
    return EvaluateCutValueDifferentversion(x,G) #returns a float

devoneshot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = 1)

Oneshotqcircuit = qml.QNode(QAOAreturnPauliZExpectation,devoneshot,interface = "torch",diff_method="parameter-shift") #Can change the diff.method during the creation of the Qnode
#Perform steps for optimization
iterations = 20
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
print(f'Initial Parameters: \nGamma = {gammas} \nBeta = {betas}')
for i in range(iterations):
    #Change the weights of the Neural Network
    opt.zero_grad()
    loss = customcost(gammas,betas,G,Oneshotqcircuit,model)
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 5 == 0:
        print(f'Current progress: {i}/{iterations}, Current approximation ratio: {-1*loss.item()/clEnergy}')
        print(f'The gradient of beta: {betas.grad}')
print(f'Final Parameters: \n Gamma = {gammas} \n Beta = {betas}')

#Train only the QAOA circuit again without the influence of the Neural Network

iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
print(f'Initial Parameters: \nGamma = {gammas} \nBeta = {betas}')
for i in range(iterations):
    opt.zero_grad()
    #print(f'Gammas = {gammas}, Betas = {betas}')
    loss = qcircuit(gammas,betas,G = G,cost = cost_h) 
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 20 == 0:
        print(f'Current progress: {i}/{iterations}, Current approximation ratio: {-1*loss.item()/clEnergy}') 
print(f'Final Parameters: \n Gamma = {gammas} \n Beta = {betas}')