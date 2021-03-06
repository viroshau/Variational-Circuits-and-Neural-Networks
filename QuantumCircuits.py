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

def QAOAreturnSamples2(parameters,G):
    """Generate computational basis state measurements instead using a shot-based device

    Args:
        gammas ([1D array]): [Array of length p with all the gammas]
        betas ([1D arry]): [Array of length p with all betas]
        G ([nx.graph]): [The graph of which the QAOA procedure is performed on]

    Returns:
        [2D array]: [Array of shape (shots,wires)]
    """
    QAOAcircuit(parameters[:len(parameters)//2],parameters[len(parameters)//2:],G)
    return qml.sample(op = None,wires = range(len(G.nodes)))

def QAOAreturnProbs(gammas,betas,G):
    QAOAcircuit(gammas,betas,G)
    return qml.probs(wires = range(len(G.nodes)))

def QAOAreturnPauliZExpectation(gammas,betas,G):
    QAOAcircuit(gammas,betas,G)
    return [qml.expval(qml.PauliZ(wires = i)) for i in range(len(G.nodes))]

def changingNNWeights(initial_weights,iterations,i,g,x = 150):
    return(1-g(i,iterations,x))*initial_weights + g(i,iterations,x)*torch.diag(torch.ones(len(initial_weights)))

def g_t(i,iterations,x):
    #creates the time-dependent scaling function t/T
    return i/(iterations-1)

def g_heaviside(i,iterations, x):
    if i < x:
        return 0
    else:
        return 1

"""def performScipyOptimizationProcedure(init_params,cost_h,nqubits,G):
    dev2 = qml.device('default.qubit', wires=nqubits) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
    
    @qml.qnode(dev2)
    def circuit(x,cost_h):
        for i in range(nqubits):
            qml.Hadamard(wires = i)
        
        p = len(x)//2
        for i in range(p):
            for j,k in G.edges():
                qml.CNOT(wires = [j,k])
                qml.RZ(-x[:p][i], wires = k)
                qml.CNOT(wires = [j,k])

            #for j,k,w in edges:
                #qml.CNOT(wires = [int(j.item()),int(k.item())])
                #qml.RZ(-x[:p][i], wires = int(k.item()))
                #qml.CNOT(wires = [int(j.item()),int(k.item())])

            for j in range(nqubits):
                qml.RX(2*x[p:][i],wires = j)

        return qml.expval(cost_h.item())

    optimizer = minimize(circuit, init_params, args = (cost_h), method='BFGS', jac = qml.grad(circuit, argnum=0))
    return optimizer"""

def CalculateProbsUsingClassicalCostFunction(gammas,betas,G,probcircuit,adjacencymatrix,configurations):
    probs = probcircuit(gammas,betas,G) #size = (2^{len(G.nodes)})
    energiesOfConfigurations = EvaluateCutOnDataset(configurations,adjacencymatrix)*probs #weighs the energies of each configurations with their probabilities
    return torch.sum(energiesOfConfigurations) #Returns the weighted sum of energies using the probabilities of obtaining each output string

def QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,optimizer,G,cost_h,clEnergy,configurations):
    #print('VQC circuit optimization: \n ------------------------------------- \n')
    #print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
    results = np.zeros(iterations)
    for i in range(iterations):
        optimizer.zero_grad()
        loss = qcircuit(gammas,betas,G,cost_h) #CalculateProbsUsingClassicalCostFunction(gammas,betas,G,qcircuit,adjacencymatrix,configurations) 
        loss.backward()
        optimizer.step()

        #Print status of the simulation
        #if i % 10 == 0:
        #    print(f'Current progress: {i}/{iterations}, Current Energy: {1*loss.item()}') 
        results[i] = loss.item()
    #print(f'\nFinal Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
    return results

def NN_Optimization(gammas,betas,tot_epoch,iterationsNN,G,NUMSHOTS,model,samplingqcircuit,adjacencymatrix,clEnergy,optimizer):
    #print('Training NN only using circuit as sample generator: \n ------------------------------------- \n')
    results = np.zeros(tot_epoch)
    for epoch in range(tot_epoch):
        #For each epoch, create a new set of samples from the VQC
        samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes)))
        samples = (2*samples-1) 
        samples = samples.type(torch.float32)

        for i in range(iterationsNN):
            #For each of the samples, perform a model prediction
            predictions = model(samples)
            #COMMENT: softmax then argmax
            #Read through simple codes on image classification and see how they do the loss function
            
            #Calculate the loss on the dataset
            loss = torch.mean(EvaluateCutOnDataset(predictions,adjacencymatrix))

            #Perform backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Print status of the simulation
        #if i % 10 == 0:
        #    print(f'Epoch: {epoch}/{tot_epoch}. Current progress: {i}/{iterationsNN}, Current Energy: {1*loss.item()}') 
        results[epoch] = loss.item()
    return results

def createCostHamiltonian(G,adjacencymatrix):
    obs = []
    coeffs1 = []
    for i,j,w in G.edges(data = True):
        coeffs1.append(0.5*w['weight'])
        obs.append(qml.PauliZ(wires = i)@qml.PauliZ(wires = j))
    coeffs1.append(-0.25*torch.sum(adjacencymatrix,axis = (0,1)).item())
    obs.append(qml.Identity(wires = 0))
    cost_h = qml.Hamiltonian(coeffs1,obs)
    return cost_h
#either use 10 shots or train 10 times. Check what happens. Most stable to average over 100 shots. 
#Coverance matrix (will blow up NN) (maybe don't do it)