import torch
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
        qobj = qiskit.assemble(t_qc,shots = self.shots,parameter_binds = [{self.parameters:parameters[0].tolist()}])
        job = self.backend.run(qobj,memory = True)
        result = job.result().get_counts()

        probs = torch.Tensor(list(result.values()))/self.shots
        stateslist = torch.Tensor([[int(i) for i in j] for j in result.keys()])
        return probs,stateslist

class HybridFunction(torch.autograd.Function):

    #Note that the neural network is a global model
    #one also sets the device within this model
    @staticmethod
    def forward(ctx,parameters):
        if not hasattr(ctx, 'QiskitCirc'):
            backend = qiskit.Aer.get_backend('aer_simulator')
            ctx.QiskitCirc = QuantumCircuit(backend = backend, shots=5000, p = len(parameters[0])//2)
        probs, samples = ctx.QiskitCirc.run(parameters)
        samples = torch.sign(model(2*samples-1)) #Pass the bitstrings through the neural network
        energies = probs*EvaluateCutOnDataset(samples,adjacencymatrix) #calculate_Cut
        ctx.save_for_backward(parameters)
        return torch.Tensor([torch.sum(energies)])

    @staticmethod
    def backward(ctx,grad_output):
        epsilon = 0.01

        parameters = ctx.saved_tensors #Retrieve the parameters used in the forward call
        shiftedparameters = torch.clone(parameters[0]) #p-long list containing all parameters
        gradients = torch.zeros(len(shiftedparameters[0]))
        #replace for-loop with parallell (joblib)
        for i in range(len(gradients)):
            #Right side
            shiftedparameters[0][i] += epsilon #Shift the i-th component of the parameter by pi/2
            probsright,samplesright = ctx.QiskitCirc.run(shiftedparameters)
            samplesright = torch.sign(model(2*samplesright-1))
            energiesRight = torch.sum(probsright*EvaluateCutOnDataset(samplesright,adjacencymatrix))

            #Left side
            shiftedparameters[0][i] -= 2*epsilon #Shift the i-th component of the parameter by -pi/2
            probsleft,samplesleft = ctx.QiskitCirc.run(shiftedparameters)
            samplesleft = torch.sign(model(2*samplesleft-1))
            energiesLeft = torch.sum(probsleft*EvaluateCutOnDataset(samplesleft,adjacencymatrix))

            #Calculate the gradient
            gradient = 0.5*(energiesRight - energiesLeft)/epsilon
            gradients[i] = gradient
            shiftedparameters[0][i] +=epsilon #Reset the shifted parameters to the original parameters for the  next for-iteration
        return torch.reshape(gradients,shape = (1,len(gradients)))

model = OneLayerNN(len(G.nodes),len(G.nodes)) #This is a global model used throughout the script
model.set_custom_weights(model,torch.eye(len(G.nodes)))

#paramvector = torch.tensor([[0.1,0.3,0.5,0.1,0.5,0.2,0.7,0.4,0.6,0.2,0.1,0.6,0.4,0.3,0.1,0.2]],requires_grad = True) #Initalize the parametervector of length 2p
qc = HybridFunction.apply

n_iters = 100
p_max = 8 

for p in range(5,p_max+1):
    parameterslist = np.random.uniform(low = 0, high = 2*np.pi, size = (n_iters,2*p))
    
    initialEnergies = np.zeros(n_iters)
    finalEnergies = np.zeros(n_iters)
    for k in tqdm(range(n_iters)):
        paramvector = torch.tensor([parameterslist[k]],requires_grad = True)
        model = OneLayerNN(len(G.nodes),len(G.nodes)) #This is a global model used throughout the script
        model.set_custom_weights(model,torch.eye(len(G.nodes)))
        qc = HybridFunction.apply
        #---------- Initial QAOA optimization -------------

        opt = torch.optim.Adam([paramvector], lr=0.01)
        initalEpochs = 150

        #loss_list = np.zeros(initalEpochs)
        for i in (range(initalEpochs)):
            opt.zero_grad()
            y1 = qc(paramvector)
            y1.backward()
            opt.step()
            #loss_list[i] = y1.item()

        initialEnergies[k] = y1.item()
        #plt.plot(loss_list)
        #plt.show()

        #---------- Train NN using circuit as sample generator -------------

        backend = qiskit.Aer.get_backend('aer_simulator')
        QiskitCirc = QuantumCircuit(backend = backend, shots=5000, p = len(paramvector[0])//2)

        optNN = torch.optim.Adam(model.parameters(),lr = 0.1)

        NN_epoch = 100

        def cost(samples,probs):
            samples =  model(2*samples-1)
            return  torch.sum(probs*EvaluateCutOnDataset(samples,adjacencymatrix))

        #loss_listNN = np.zeros(NN_epoch)

        for i in (range(NN_epoch)):
            optNN.zero_grad()
            probs,samples = QiskitCirc.run(paramvector)
            loss = cost(samples,probs)
            loss.backward()
            optNN.step()
            #loss_listNN[i] = loss.item()

        #plt.plot(loss_listNN)

        #---------- Train QAOA parameters with heaviside NN -------------

        opt = torch.optim.Adam([paramvector], lr=0.01)

        initHeavisideEpoch = 20

        #loss_list2 = np.zeros(initHeavisideEpoch)
        for i in (range(initHeavisideEpoch)):
            opt.zero_grad()
            y1 = qc(paramvector)
            y1.backward()
            opt.step()
            #loss_list2[i] = y1.item()

        finalEpochs = 150

        model.set_custom_weights(model,torch.eye(len(G.nodes))) #Change the model-weights into a diagonal with ones

        #loss_list3 = np.zeros(finalEpochs)
        for i in (range(finalEpochs)):
            opt.zero_grad()
            y1 = qc(paramvector)
            y1.backward()
            opt.step()
            #loss_list3[i] = y1.item()
        finalEnergies[k] = y1.item()
        #plt.plot(np.concatenate((loss_list2,loss_list3)))

        #---------- Train QAOA parameters without NN -------------

        """opt = torch.optim.Adam([paramvector], lr=0.01)

        num_epoch = 20

        loss_list = np.zeros(num_epoch)
        for i in (range(num_epoch)):
            opt.zero_grad()
            y1 = qc(paramvector)
            y1.backward()
            opt.step()
            loss_list[i] = y1.item()

        finalEnergies[k] = y1.item()
        #plt.plot(loss_list)"""


        #plt.show()
    energies = np.zeros((2,len(initialEnergies)))
    energies[0] = initialEnergies
    energies[1] = finalEnergies
    savestring = f'/ADAM,p={p},init:{initalEpochs},NN:{NN_epoch},heaviside:{initHeavisideEpoch},final_iterations:{finalEpochs}'

    np.save(foldername+savestring,energies)