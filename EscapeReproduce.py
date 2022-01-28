from QuantumCircuits import *


def customcost(gammas,betas,G,probcircuit,neuralNet,adjacencymatrix,configurations):
    probs = probcircuit(gammas,betas,G) #size = (2^{len(G.nodes)})
    x = torch.sign(neuralNet(configurations)) #The predictions from the neural network based on the output. Note that configurations is a list with all possible bitstring outcomes from the quantum system
    energiesOfConfigurations = EvaluateCutOnDataset(x,adjacencymatrix)*probs #weighs the energies of each configurations with their probabilities
    return torch.sum(energiesOfConfigurations) #Returns the weighted sum of energies using the probabilities of obtaining each output string

#Define graph and and cost Hamiltonian
G = CreateRegularGraph(5,4,True)
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,left,right = BestClassicalHeuristicResult(G)
cost_h = createCostHamiltonian(G,adjacencymatrix)
#cost_h, mixer_h = qml.qaoa.maxcut(G)

configs = torch.tensor(CreateBinaryList(len(G.nodes)),dtype = torch.float32)
print('Energies: ')
print(EvaluateCutOnDataset(2*configs-1,adjacencymatrix))
print(torch.argmin(EvaluateCutOnDataset(2*configs-1,adjacencymatrix)))

#Define Devices to run the circuits
NUMSHOTS = 500
devExact = qml.device('default.qubit.torch', wires = len(G.nodes)) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
devShot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = NUMSHOTS)

costHamiltonianCircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best') #Returns the cost value after running the circuit
probabilityCircuit = qml.QNode(QAOAreturnProbs,devExact,interface = "torch") #Returns the state probability before measurement
shotCircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch') #Returns a set of measurement samples

#Turn gammas and betas into trainable variables
gammas = torch.tensor([0.1,2,3,4,6,7],dtype = torch.float64)
gammas = torch.autograd.Variable(gammas,requires_grad = True)
betas = torch.tensor([0.1,2,3,4,9,2],dtype = torch.float64)
betas = torch.autograd.Variable(betas,requires_grad = True)

# ------------- Initial Optimization of VQC parameters ------------- 
iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)

QAOA_OptimizationWithoutNN(gammas,betas,iterations,costHamiltonianCircuit,opt,G,cost_h,clEnergy,2*configs-1)

# ------------- Train the NN using VQC circuit as sample generator ------------- 

#Sample form the circuit to generate the strings that the NN should be trained with

samplingqcircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch',diff_method = 'best')
samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes))) #Returns the samples from the circuit of shape (NUMSHOTS,numqubits)

#Train the Neural Network
model = OneLayerNN(len(G.nodes),len(G.nodes))

tot_epoch = 25
iterationsNN = 1
opt = torch.optim.Adam(model.parameters(),lr = 0.3)

NN_Optimization(gammas,betas,tot_epoch,iterationsNN,G,NUMSHOTS,model,samplingqcircuit,adjacencymatrix,clEnergy,opt)

# ------------- Train QAOA using varying NN parameters ------------- 

#Perform steps for optimization
iterations = 50
opt = torch.optim.Adam([gammas,betas],lr = 0.3)

initialweights = model.return_weights() #Used when the NN landscape is gradually changed.

print('Training VQC using a changing Neural Network: \n ------------------------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):
    #Change the weights of the Neural Network

    weights = changingNNWeights(initialweights,iterations,i,g_t)
    model.set_custom_weights(model,weights)

    #Perform optimization steps
    opt.zero_grad()
    loss = customcost(gammas,betas,G,probabilityCircuit,model,adjacencymatrix,2*configs-1)
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 10 == 0:
        print(f'Current progress: {i}/{iterations}, Current Energy: {1*loss.item()}')
print(f'Final Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

# ------------- Final training QAOA without NN ------------- 

iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
QAOA_OptimizationWithoutNN(gammas,betas,iterations,costHamiltonianCircuit,opt,G,cost_h,clEnergy,2*configs-1)

#QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,opt,G,cost_h,clEnergy)

fig, axs = plt.subplots(2, sharex=True)
axs[0].hist(list(range(2**len(G.nodes))),weights = probabilityCircuit(gammas,betas,G).detach().numpy(),bins = 2**len(G.nodes),label = 'Probs from algo',color = 'y')
axs[1].hist(list(range(2**len(G.nodes))),weights = -1*EvaluateCutOnDataset(2*configs-1,adjacencymatrix).detach().numpy(),bins = 2**len(G.nodes),label = 'Energy distribution')
axs[1].set(xlabel = 'Bitstrings')
fig.legend(loc = 'center',ncol = 2)
plt.show()


#Coverance matrix (will blow up NN) (maybe don't do it)