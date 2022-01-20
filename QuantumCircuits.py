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
    return qml.sample(op = None,wires = G.nodes)

def QAOAreturnProbs(gammas,betas,G):
    QAOAcircuit(gammas,betas,G)
    return qml.probs(wires = G.nodes)

def listofBinaryToInt(binarylist):
    descimalnumber = 0
    for i in range(len(binarylist)):
        if binarylist[len(binarylist)-i-1] == 1:
            descimalnumber += 2**i
    return descimalnumber

def otherwaytranslation(binarylist):
    descimalnumber = 0
    for i in range(len(binarylist)):
        if binarylist[i] == 1:
            descimalnumber += 2**i
    return descimalnumber

#Define graph
G = CreateRegularGraph(8,5)
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
iterations = 100
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

#Check if the reshaping actually makes any sense by comparing it to the probability distribution from the circuit. 
""" 
samplesdesimal = torch.zeros(NUMSHOTS)
for i in range(NUMSHOTS):
    samplesdesimal[i] = listofBinaryToInt(samples[i])

#Count the number of times a bitstring appears when sampling, using the integerversion of the bitstring as key and number of appearences as value. 
counts = {i:0 for i in range(2**len(G.nodes))} #Initialize keys to have zero counts
for i in samplesdesimal.tolist():
  counts[int(i)] = counts.get(int(i), 0) + 1 #Count the number of occurences

sortedCounts = dict(sorted(counts.items()))

qcirc = qml.QNode(QAOAreturnProbs,devExact,interface = 'torch',diff_method = 'best')
probabilities = qcirc(gammas,betas,G)
probs = probabilities.detach().numpy()
prob_dic = {i:probs[i]*NUMSHOTS for i in range(2**len(G.nodes))}

plt.figure()
plt.hist(list(range(2**len(G.nodes))),bins = 2**len(G.nodes),weights = probs,alpha = 0.5,label = 'probability')
plt.hist(list(sortedCounts.keys()),bins = 2**len(G.nodes),weights = np.array(list(sortedCounts.values()))/NUMSHOTS,alpha = 0.4,label = 'samples')
plt.legend()
plt.show()
"""

samples = (2*samples-1)
samples = samples.type(torch.float32) #Cast the samples into float
print(samples.type())
average_cost = 0
for i in range(len(samples)):
    #print(f'The cost: {EvaluateCutValueDifferentversion(G,samples[i])}, the sample: {samples[i]}')
    average_cost += EvaluateCutValueDifferentversion(samples[i],G)

print(average_cost/NUMSHOTS)
#Train NN
model = torch.nn.Sequential(
    torch.nn.Linear(len(G.nodes),len(G.nodes)),
    torch.nn.Tanh()
    )

iterationsNN = 1
opt = torch.optim.Adam(model.parameters(),lr = 0.3)

for i in range(iterationsNN):
    #Create samples and predict on them
    samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes)))
    samples = (2*samples-1)
    samples = samples.type(torch.float32)
    predictions = model(samples)
    
    #Calculate the loss on the dataset
    loss = EvaluateCutOnDataset(predictions,G) 

    #Perform backprop
    opt.zero_grad()
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 20 == 0:
        print(f'Current progress: {i}/{iterationsNN}, Current approximation ratio: {-1*loss.item()/clEnergy}') 

print(model(samples))