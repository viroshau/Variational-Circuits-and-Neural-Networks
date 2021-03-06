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
#clEnergy,left,right = BestClassicalHeuristicResult(G)

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

"""
def CostLandscapeHeatMap(meshGamma,meshBeta,Z,p):
    fig = plt.figure()
    c = plt.pcolormesh(meshGamma,meshBeta,Z,cmap = 'coolwarm')
    plt.ylabel(r'$\beta_p$')
    plt.xlabel(r'$\gamma_p$')
    #plt.xlim(0,np.pi)
    #plt.ylim(0,np.pi/2)
    plt.title(f'Cost-Landscape for a {len(G.nodes())} nodes u{3}R graph, p = {p}')
    plt.colorbar(c)
    plt.show()  

def CostLandscapeplotter(meshGamma,meshBeta,Z, p = 0):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshGamma,meshBeta,Z,cmap='coolwarm')
    ax.set_ylabel(r'$\beta_p$')
    ax.set_xlabel(r'$\gamma_p$')
    ax.set_zlabel('C')
    ax.set_title(f'Depth p = {p}')
    #ax.scatter(0.58821361, 0.36506835,np.min(Z),s = 150,marker = 'D',c = 'darkorange')
    #ax.scatter(0.64220579,1.14829102,np.max(Z),s = 150,marker = 'D',c = 'seagreen')
    plt.show()  

@np.vectorize
def CalculateCostlandscape(gamma,beta):
    params = torch.tensor([gamma,beta])
    probsright,samplesright = circuit.run(params)
    samplesright = torch.sign(model(2*samplesright-1))
    costs = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))
    costs1 = torch.sum(circuit.shots*probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))
    costs2 = torch.sum(circuit.shots*probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix)**2)
    variance = costs2/(circuit.shots-1) - costs1**2/(circuit.shots*(circuit.shots-1))

    return costs.item(),variance.item()

def CostLandscapeProcedureFull(gammabound,betabound,discretization,p):
    gammas = np.arange(*gammabound,discretization)
    betas = np.arange(*betabound,discretization)

    meshGamma,meshBeta = np.meshgrid(gammas, betas)

    start_time = time.time()
    Z,V = CalculateCostlandscape(meshGamma,meshBeta)
    end_time = time.time()
    print(f"The execution time of Meshgrid-calcs is: {end_time-start_time}")
    #CostLandscapeplotter(meshGamma,meshBeta,Z,p)
    return meshGamma,meshBeta,Z,V

simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuit(simulator,5000,1)

gammabound = (0,2*np.pi)
betabound = (0,2*np.pi)
model = OneLayerNN(len(G.nodes),len(G.nodes))
model.set_custom_weights(model,torch.eye(len(G.nodes)))
meshGamma,meshBeta,Z,V = CostLandscapeProcedureFull(gammabound,betabound,0.1,1)
CostLandscapeHeatMap(meshGamma,meshBeta,Z, p = 1)
CostLandscapeHeatMap(meshGamma,meshBeta,V, p = 1)
"""

model = OneLayerNN(len(G.nodes),len(G.nodes))
simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuit(simulator,50000,1)
model.set_custom_weights(model,torch.eye(len(G.nodes)))


gammas = torch.linspace(0,2*np.pi,100)
results = torch.zeros(len(gammas))
gradients = torch.zeros(len(gammas))
gradients2 = torch.zeros(len(gammas))
for i in tqdm(range(len(gammas))):
    rightshift = torch.tensor([[gammas[i].item(),0.4]])
    probsright,samplesright = circuit.run(rightshift[0])
    samplesright = torch.sign(model(2*samplesright-1))
    energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))
    results[i] = energiesRight
    gradients[i] = parameter_shift_otherversion(rightshift,circuit,0.01)[0][0]
    gradients2[i] = parameter_shift_otherversion(rightshift,circuit,epsilon = 0.1)[0][0]

plt.plot(gammas.detach().numpy(),results.detach().numpy(),label = 'mean_value')
plt.plot(gammas.detach().numpy(),gradients.detach().numpy(),label = '0.01')
plt.plot(gammas.detach().numpy(),gradients2.detach().numpy(),label = '0.1')
plt.legend()
plt.show()

stop
simulator = qiskit.Aer.get_backend('aer_simulator')
circuit = QuantumCircuit(simulator,50000,8)

model = OneLayerNN(len(G.nodes),len(G.nodes))
model.set_custom_weights(model,torch.eye(len(G.nodes)))

parameters = torch.tensor([[1.0,1.0,1.0,2.0,9.0,8.0,6,3,8.3,9.2,0.1,9.3,7.9,8.0,4.7,5.5]])
opt = torch.optim.Adam([parameters], lr=0.01)

losses = np.zeros(400)
for i in tqdm(range(400)):
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