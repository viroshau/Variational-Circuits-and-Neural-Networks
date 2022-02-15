from joblib import Parallel, delayed
import time as time
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit.visualization import *
from graphCreation import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

seed = 51
foldername = f'./SHOTBASED5NodeESCAPEResultsSameGraph{seed}'
np.random.seed(seed) #Use seed to generate a random graph
G = CreateRegularGraph(5,4,True,seed = seed)
#G = CreateGraphInstanceB(16,3,seed = seed)
np.random.seed() #Remove the random seed when generating initial points, etc
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,left,right = BestClassicalHeuristicResult(G)

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
    gradients = torch.Tensor(Parallel(n_jobs=4)(delayed(param_shift_i) (i,leftshift[i],rightshift[i]) for i in range(len(parameters[0]))))
    #gradients = torch.zeros(len(parameters[0]))
    #for i in range(len(gradients)):
    #    gradients[i] = param_shift_i(i,leftshift[i],rightshift[i])
    return torch.reshape(gradients,shape = (1,len(gradients)))/epsilon

simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuit(simulator,50000,8)

model = OneLayerNN(len(G.nodes),len(G.nodes))
model.set_custom_weights(model,torch.eye(len(G.nodes)))
parameters = torch.tensor([[1.0,1.0,1.0,2.0,9.0,8.0,6,3,8.3,9.2,0.1,9.3,7.9,8.0,4.7,5.5]])
opt = torch.optim.Adam([parameters], lr=0.01)

losses = np.zeros(100)
for i in tqdm(range(100)):
    opt.zero_grad()
    parameters.grad = parameter_shift_otherversion(parameters,circuit)
    opt.step()
    probsright,samplesright = circuit.run(parameters[0])
    samplesright = torch.sign(model(2*samplesright-1))
    energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))
    losses[i] = energiesRight.item()

plt.figure()
plt.plot(losses)
plt.show()
"""print(opt)
print(parameters.grad)
parameters.grad = torch.tensor([[1.0,1.0,1.0,1.0]])
print(parameters.grad)"""