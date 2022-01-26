from mimetypes import init
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import networkx as nx
import matplotlib.pyplot as plt

def CreateRegularGraph(n,degree,randomweights = False,seed = None):
    G = nx.generators.random_regular_graph(degree,n,seed = seed)
    if randomweights == True:
        for i,j in G.edges:
            G[i][j]['weight'] = np.random.rand() #Assign edge weights to be a float between [0,1]
    else:
        for i,j in G.edges:
            G[i][j]['weight'] = 1 #Assign weights to be 1 if you don't want weights
    return G

def CreateAdjacencyMatrix(G):
    """Returns the adjacency matrix as a numpy 2D matrix. This utilizes for-loops, however is only called once so it's fine

    Args:
        G ([nx.graph]): [The graph instance that we want the adjacecy matrix of]

    Returns:
        [np.ndarray(len(G.nodes),len(G.nodes))]: [A 2D numpy array containing the edges of the matrix]
    """
    adjacencymatrix = np.zeros((len(G.nodes),len(G.nodes)))
    for i,j,w in G.edges(data = True):
        adjacencymatrix[i,j] = w['weight'] 
        adjacencymatrix[j,i] = w['weight']
    return torch.tensor(adjacencymatrix,dtype = torch.float32)

def EvaluateCutValue(adjacencymatrix,x):
    """Evaluates the value of the cut given a bitstring x on graph G

    Args:
        G ([nx.graph]): [Graph instance to consider]
        x ([np.array): [1D array containing the bitstring at each position. The elements are either 1 or -1]
    
    Return:
        ([float]): The value of the cut given the configuration.
    """
    return -1*(0.25*torch.sum(adjacencymatrix,dim = (0,1)) - 0.25*x@adjacencymatrix@x)

def EvaluateCutOnDataset(dataset,adjacencymatrix):
    """Given a dataset filled with strings of type [1,-1,1,1,-1,-1] etc, this functions will evalute the mean cost of the set of strings

    Args:
        dataset ([torch(N_samples,len(G.nodes))]): [The dataset of pauli-Z eigenvalue strings]
        adjacencymatrix ([type]): [The adjacency matrix of graph G]

    Returns:
        [float]: [Returns the mean cost of all N_samples in the dataset]
    """
    y = torch.matmul(adjacencymatrix, dataset.T).T #Perform adjacent multiplication on all vectors in dataset. returnsize = (N_samples,nodes)
    xy = (torch.sum(dataset*y,dim = 1)) # perform xAx on all N_samples
    result = 0.25*xy - 0.25*torch.sum(adjacencymatrix,dim = (0,1))
    return torch.mean(result)

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
    result_matrix = (1-g(i,iterations))*initial_weights + g(i,iterations)*torch.diag(torch.ones(len(initial_weights)))
    return result_matrix

def g_t(i,iterations):
    #creates the time-dependent scaling function t/T
    return i/(iterations-1)

#Define graph
G = CreateRegularGraph(8,3)
adjacencymatrix = CreateAdjacencyMatrix(G)
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
iterations = 10
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
print('Initial VQC circuit optimization: \n ------------------------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):
    opt.zero_grad()
    #print(f'Gammas = {gammas}, Betas = {betas}')
    loss = qcircuit(gammas,betas,G = G,cost = cost_h) 
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 20 == 0:
        print(f'Current progress: {i}/{iterations}') 
print(f'Final Parameters: \n Gamma = {gammas.data.numpy()} \n Beta = {betas.data.numpy()}')

#Sample form the circuit to generate the strings that the NN should be trained with

samplingqcircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch',diff_method = 'best')
samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes))) #Returns the samples from the circuit of shape (NUMSHOTS,numqubits)

#Define the NN
class OneLayerNN(nn.Module):
    def __init__(self,D_in,D_out):
        super(OneLayerNN,self).__init__()
        self.linear = torch.nn.Linear(D_in,D_out,bias = False)
        self.tanh = nn.Tanh() #[-1,1]
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

#Train the Neural Network
model = OneLayerNN(len(G.nodes),len(G.nodes))

tot_epoch = 2
iterationsNN = 20
opt = torch.optim.Adam(model.parameters(),lr = 0.3)
print('Training NN only using circuit as sample generator: \n ------------------------------------- \n')

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
        loss = EvaluateCutOnDataset(predictions,adjacencymatrix) 

        #Perform backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        #Print status of the simulation
        if i % 10 == 0:
            print(f'Epoch: {epoch}/{tot_epoch}. Current progress: {i}/{iterationsNN}') 

#Train variational circuit where the outputs from the circuit is passed through a neural network. 

#Do this shots (10)-number of times in parallell, then backprop. Currently it does not do so in parallell
def customcost(gammas,betas,G,qcircuit,neuralNet,adjacencymatrix):
    cost = 0
    for i in range(10):
        x = (qcircuit(gammas,betas,G)).float() #one shot to the circuit #SWITCH THE CIRCUIT WITH A SAMPLE-VERSION INSTEAD, calculate cost, then average.
        x = neuralNet(x) #pass it through the neural network
        cost += EvaluateCutValue(adjacencymatrix,x)/10
    return cost #returns a float

devoneshot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = 1)
#either use 10 shots or train 10 times. Check what happens. Most stable to average over 100 shots. 

Oneshotqcircuit = qml.QNode(QAOAreturnPauliZExpectation,devoneshot,interface = "torch",diff_method="parameter-shift") #One-shot based Qnode where the return value is the PauliZ value at each qubit

print(Oneshotqcircuit(gammas,betas,G))
#Perform steps for optimization
iterations = 20
opt = torch.optim.Adam([gammas,betas],lr = 0.3)

initialweights = model.return_weights() #Used when the NN landscape is gradually changed.
#Coverance matrix (will blow up NN) (maybe don't do it)

print('Training VQC using a changing Neural Network: \n ------------------------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):
    #Change the weights of the Neural Network

    weights = changingNNWeights(initialweights,iterations,i,g_t)
    model.set_custom_weights(model,weights)

    #Perform optimization steps
    opt.zero_grad()
    loss = customcost(gammas,betas,G,Oneshotqcircuit,model,adjacencymatrix)
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 1 == 0:
        print(f'Current progress: {i}/{iterations}')
        print(f'The gradient of Gamma: {gammas.grad}')
        print(f'The gradient of Beta: {betas.grad}')
print(f'Final Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

#Train only the QAOA circuit again without the influence of the Neural Network

iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
print('Final Training procedure without a Neural Network: \n ------------------------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):
    opt.zero_grad()
    loss = qcircuit(gammas,betas,G = G,cost = cost_h) 
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 20 == 0:
        print(f'Current progress: {i}/{iterations}') 
print(f'Final Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')