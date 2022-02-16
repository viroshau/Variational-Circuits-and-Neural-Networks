import numpy as np
import qiskit
from joblib import Parallel, delayed
import time as time
import torch
import networkx as nx

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def CreateRenyiGraph(n,p,randomweights = False,seed = None):
    G = nx.fast_gnp_random_graph(n,p,seed = seed)
    if randomweights == True:
        for i,j in G.edges:
            G[i][j]['weight'] = np.random.rand() #Assign edge weights to be a float between [0,1]
    else:
        for i,j in G.edges:
            G[i][j]['weight'] = 1 #Assign weights to be 1 if you don't want weights
    return G

def CreateRegularGraph(n,degree,randomweights = False,seed = None):
    G = nx.generators.random_regular_graph(degree,n,seed = seed)
    if randomweights == True:
        for i,j in G.edges:
            G[i][j]['weight'] = np.random.standard_normal() #Assign edge weights to be a float between [0,1]
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

def EvaluateCutOnDataset(dataset,adjacencymatrix):
    """Given a dataset filled with strings of type [1,-1,1,1,-1,-1] etc, this functions will evalute the mean cost of the set of strings

    Args:
        dataset ([torch(N_samples,len(G.nodes))]): [The dataset of pauli-Z eigenvalue strings]
        adjacencymatrix ([type]): [The adjacency matrix of graph G]

    Returns:
        [ndarray]: [Returns the cost of all N_samples in the dataset]
    """
    y = torch.matmul(adjacencymatrix, dataset.T).T #Perform adjacent multiplication on all vectors in dataset. returnsize = (N_samples,nodes)
    xy = (torch.sum(dataset*y,dim = 1)) # perform xAx on all N_samples
    result = 0.25*xy - 0.25*torch.sum(adjacencymatrix,dim = (0,1)) 
    return result 

seed = 51
foldername = f'./SHOTBASED5NodeESCAPEResultsSameGraph{seed}'
np.random.seed(seed) #Use seed to generate a random graph
G = CreateRegularGraph(5,4,True,seed = seed)
#G = CreateGraphInstanceB(16,3,seed = seed)
np.random.seed() #Remove the random seed when generating initial points, etc

print('-----hello!------')
print(G)

adjacencymatrix = CreateAdjacencyMatrix(G)
#clEnergy,left,right = BestClassicalHeuristicResult(G)


class OneLayerNN(torch.nn.Module):
    def __init__(self,D_in,D_out):
        super(OneLayerNN,self).__init__()
        self.linear = torch.nn.Linear(D_in,D_out,bias = False)
        self.tanh = torch.nn.Tanh() #[-1,1]  #Potentially look into using hardtanh instead
        
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

def problemUnitary(parameters,k,G,qc):
    for i,j,w in G.edges(data = True):
        qc.rzz(parameters[k]*w['weight'],i,j)
    qc.barrier()

def mixerUnitary(parameters,k,G,qc):
    for i in range(0,len(G.nodes)):
        qc.rx(parameters[k],i)

def constructQAOACircuit(G,parameters):
    qc = qiskit.QuantumCircuit(len(G.nodes))
    qc.h([i for i in range(len(G.nodes))])
    for k in range(len(parameters)//2):
        problemUnitary(parameters,2*k,G,qc)
        mixerUnitary(parameters,2*k+1,G,qc)
    qc.measure_all()
    return qc

class QuantumCircuit:

    def __init__(self,backend,shots,p):
        self.parameters = qiskit.circuit.ParameterVector('params',2*p)
        self._circuit = constructQAOACircuit(G,self.parameters)
        #------------------

        self.backend = backend
        self.shots = shots

    def run(self,parameters):
        t_qc = qiskit.transpile(self._circuit,self.backend)
        qobj = qiskit.assemble(t_qc,shots = self.shots,parameter_binds = [{self.parameters:parameters.tolist()}])
        job = self.backend.run(qobj,memory = True)
        result = job.result().get_counts()

        probs = torch.Tensor(list(result.values()))/self.shots
        stateslist = torch.Tensor([[int(i) for i in j] for j in result.keys()])
        return probs,stateslist

@torch.no_grad()
def compute_parameter_shift(parameters,circuit):
    epsilon = np.pi/2

    shiftedparameters = torch.clone(parameters) #p-long list containing all parameters
    leftshift =  parameters[0].repeat(len(parameters[0]),1) - epsilon*torch.eye(len(parameters[0]))
    rightshift = parameters[0].repeat(len(parameters[0]),1) + epsilon*torch.eye(len(parameters[0]))
    gradients = torch.zeros(len(shiftedparameters[0]))
    #replace for-loop with parallell (joblib)
    for i in range(len(gradients)):
        #Right side
        shiftedparameters[0][i] += epsilon #Shift the i-th component of the parameter by pi/2
        probsright,samplesright = circuit.run(shiftedparameters[0])
        samplesright = torch.sign(model(2*samplesright-1))
        energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))

        #Left side
        shiftedparameters[0][i] -= 2*epsilon #Shift the i-th component of the parameter by -pi/2
        probsleft,samplesleft = circuit.run(shiftedparameters[0])
        samplesleft = torch.sign(model(2*samplesleft-1))
        energiesLeft = torch.sum(probsleft*EvaluateCutOnDataset(samplesleft,adjacencymatrix))

        #Calculate the gradient
        gradient = 0.5*(energiesRight - energiesLeft)
        gradients[i] = gradient
        shiftedparameters[0][i] +=epsilon #Reset the shifted parameters to the original parameters for the  next for-iteration
    return torch.reshape(gradients,shape = (1,len(gradients)))

@torch.no_grad()
def param_shift_i(i,leftshift,rightshift):
    probsright,samplesright = circuit.run(rightshift)
    samplesright = torch.sign(model(2*samplesright-1))
    energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))

    probsleft,samplesleft = circuit.run(leftshift)
    samplesleft = torch.sign(model(2*samplesleft-1))
    energiesLeft = torch.sum(probsleft*EvaluateCutOnDataset(samplesleft,adjacencymatrix))

    return 0.5*(energiesRight - energiesLeft)

def parameter_shift_otherversion(parameters,circuit,epsilon = 0.01):

    leftshift =  parameters[0].repeat(len(parameters[0]),1) - epsilon*torch.eye(len(parameters[0]))
    rightshift = parameters[0].repeat(len(parameters[0]),1) + epsilon*torch.eye(len(parameters[0]))
    gradients = torch.Tensor(Parallel(n_jobs=-1)(delayed(param_shift_i) (i,leftshift[i],rightshift[i]) for i in range(len(parameters[0]))))
    #gradients = torch.zeros(len(parameters[0]))
    #for i in range(len(gradients)):
    #    gradients[i] = param_shift_i(i,leftshift[i],rightshift[i])
    return torch.reshape(gradients,shape = (1,len(gradients)))/epsilon

model = OneLayerNN(len(G.nodes),len(G.nodes))
simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuit(simulator,50000,1)
model.set_custom_weights(model,torch.eye(len(G.nodes)))

start = time.time()
gammas = torch.linspace(0,2*np.pi,100)
results = torch.zeros(len(gammas))
gradients = torch.zeros(len(gammas))
gradients2 = torch.zeros(len(gammas))
for i in (range(len(gammas))):
    rightshift = torch.tensor([[gammas[i].item(),0.4]])
    probsright,samplesright = circuit.run(rightshift[0])
    samplesright = torch.sign(model(2*samplesright-1))
    energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))
    results[i] = energiesRight
    gradients[i] = parameter_shift_otherversion(rightshift,circuit,0.01)[0][0]
    gradients2[i] = parameter_shift_otherversion(rightshift,circuit,epsilon = 0.1)[0][0]
end = time.time()
print(f'fullfrt!, time = {end-start}')

