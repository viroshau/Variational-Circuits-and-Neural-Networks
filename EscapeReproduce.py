from QuantumCircuits import *


def customcost(gammas,betas,G,probcircuit,neuralNet,adjacencymatrix,configurations):
    probs = probcircuit(gammas,betas,G) #size = (2^{len(G.nodes)})
    x = torch.sign(neuralNet(configurations)) #The predictions from the neural network based on the output. Note that configurations is a list with all possible bitstring outcomes from the quantum system
    energiesOfConfigurations = EvaluateCutOnDataset(x,adjacencymatrix)*probs #weighs the energies of each configurations with their probabilities
    return torch.sum(energiesOfConfigurations) #Returns the weighted sum of energies using the probabilities of obtaining each output string

#Define graph
G = CreateRegularGraph(9,4)
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,_,_ = BestClassicalHeuristicResult(G)
cost_h, mixer_h = qml.qaoa.maxcut(G)

configs = torch.tensor(CreateBinaryList(len(G.nodes)),dtype = torch.float32)

#Define Devices to run the circuit
NUMSHOTS = 500
devExact = qml.device('default.qubit.torch', wires = len(G.nodes)) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
devShot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = NUMSHOTS)

#Turn gammas and betas into trainable variables
gammas = torch.tensor([0.1,2,3,4,6,7],dtype = torch.float64)
gammas = torch.autograd.Variable(gammas,requires_grad = True)
betas = torch.tensor([0.1,2,3,4,9,2],dtype = torch.float64)
betas = torch.autograd.Variable(betas,requires_grad = True)

qcircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best') #Can change the diff.method during the creation of the Qnode
probabilitycircuit = qml.QNode(QAOAreturnProbs,devExact,interface = "torch")
#Perform steps for the initial optimization
iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)

QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,opt,G,cost_h,clEnergy,2*configs-1)

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
#Coverance matrix (will blow up NN) (maybe don't do it)

print('Training VQC using a changing Neural Network: \n ------------------------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):
    #Change the weights of the Neural Network

    weights = changingNNWeights(initialweights,iterations,i,g_t)
    model.set_custom_weights(model,weights)

    #Perform optimization steps
    opt.zero_grad()
    loss = customcost(gammas,betas,G,probabilitycircuit,model,adjacencymatrix,2*configs-1)
    loss.backward()
    opt.step()

    #Print status of the simulation
    if i % 10 == 0:
        print(f'Current progress: {i}/{iterations}, Current approximation ratio: {-1*loss.item()/clEnergy}')
print(f'Final Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

#Train only the QAOA circuit again without the influence of the Neural Network

iterations = 60
opt = torch.optim.Adam([gammas,betas],lr = 0.3)
QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,opt,G,cost_h,clEnergy,2*configs-1)

#QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,opt,G,cost_h,clEnergy)

fig, axs = plt.subplots(2, sharex=True)
axs[0].hist(list(range(2**len(G.nodes))),weights = probabilitycircuit(gammas,betas,G).detach().numpy(),bins = 2**len(G.nodes),label = 'Probs from algo',color = 'y')
axs[1].hist(list(range(2**len(G.nodes))),weights = -1*EvaluateCutOnDataset(2*configs-1,adjacencymatrix).detach().numpy()/clEnergy,bins = 2**len(G.nodes),label = 'Approx ratio distribution')
axs[1].set(xlabel = 'Bitstrings')
fig.legend(loc = 'center',ncol = 2)
plt.show()

