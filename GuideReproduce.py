from QuantumCircuits import *

# ------------- Set up graphs, circuits and variables -------------  

G = CreateRegularGraph(10,4)
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,_,_ = BestClassicalHeuristicResult(G)
cost_h, mixer_h = qml.qaoa.maxcut(G)

configs = torch.tensor(CreateBinaryList(len(G.nodes)),dtype = torch.float32) #All the possible configurations for the number of available nodes on the graph. Shape = (2^numqubits,numqubits)

#Define Devices to run the circuit
NUMSHOTS = 500
devExact = qml.device('default.qubit.torch', wires = len(G.nodes)) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
devShot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = NUMSHOTS)

#Turn gammas and betas into trainable variables
gammas = torch.tensor([0.1,2,3,4,6,7],dtype = torch.float64)
gammas = torch.autograd.Variable(gammas,requires_grad = True)
betas = torch.tensor([0.1,2,3,4,9,2],dtype = torch.float64)
betas = torch.autograd.Variable(betas,requires_grad = True)

#Define the necessary quantum circuits
costHamiltonianCircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best') #Returns the cost value after running the circuit
probabilityCircuit = qml.QNode(QAOAreturnProbs,devExact,interface = "torch") #Returns the state probability before measurement
shotCircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch') #Returns a set of measurement samples

# ------------- Train QAOA and NN simultainously ------------- 

def HybridCostForQAOA(gammas,betas,G,probcircuit,neuralNet,adjacencymatrix,configurations):
    probs = probcircuit(gammas,betas,G) #size = (2^{len(G.nodes)})
    x = torch.sign(neuralNet(configurations)) #The predictions from the neural network based on the output. Note that configurations is a list with all possible bitstring outcomes from the quantum system
    energiesOfConfigurations = EvaluateCutOnDataset(x,adjacencymatrix)*probs #weighs the energies of each configurations with their probabilities
    return torch.sum(energiesOfConfigurations) #Returns the weighted sum of energies using the probabilities of obtaining each output string

def HybridCostForNN(gammas,betas,G,samplingqcircuit,neuralNet,adjacencymatrix,NUMSHOTS,alpha):
    #The cost function for the Neural network needs to also include the regularizer term.

    samples = torch.reshape(samplingqcircuit(gammas,betas,G),(NUMSHOTS,len(G.nodes))) #Reshape into correct shape
    samples = (2*samples-1) #Turn into {-1,1} bitstring
    samples = samples.type(torch.float32)
    samples = neuralNet(samples) #passes the samples through the NN which effectively produces an output
    effectivecuts = EvaluateCutOnDataset(samples,adjacencymatrix)  #Shape = (NUMSHOTS,). Gives the cut value of all samples after being passed through the NN
    cost = torch.mean(effectivecuts) + alpha * torch.sum(torch.abs(neuralNet.linear.weight-torch.eye(len(G.nodes))),dim = (0,1)) #Might need an if-statement or something like that to 
    return cost

model = OneLayerNN(len(G.nodes),len(G.nodes))

optCirc = torch.optim.Adam([gammas,betas],lr = 0.3)
optNN = torch.optim.Adam(model.parameters(),lr = 0.3)

iterations = 50
alpha = 0.3

print(f'\nInitial values:\nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}\n----------------------')
for i in range(iterations):
    
    #Train the circuit parameters 

    optCirc.zero_grad()
    lossCirc = HybridCostForQAOA(gammas,betas,G,probabilityCircuit,model,adjacencymatrix,2*configs-1)
    lossCirc.backward()
    optCirc.step()

    #Sample from the circuit and train the NN parameters

    optNN.zero_grad()
    lossNN = HybridCostForNN(gammas,betas,G,shotCircuit,model,adjacencymatrix,NUMSHOTS,alpha)
    lossNN.backward()
    optNN.step()
    if i % 10 == 0:
        print(f'Progress = {i}/{iterations}. Current approx ratio = {-1*lossCirc/clEnergy}')

print(f'\nFinal values:\nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}\n----------------------')

# ------------- Train QAOA using varying NN parameters ------------- 

iterations = 50

initialweights = model.return_weights() #Used when the NN landscape is gradually changed.

print('Training VQC using a changing Neural Network: \n ---------------------- \n')
print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')
for i in range(iterations):

    #Change the weights of the Neural Network
    weights = changingNNWeights(initialweights,iterations,i,g_t)
    model.set_custom_weights(model,weights)

    #Perform optimization steps
    optCirc.zero_grad()
    loss = HybridCostForQAOA(gammas,betas,G,probabilityCircuit,model,adjacencymatrix,2*configs-1)
    loss.backward()
    optCirc.step()

    #Print status of the simulation
    if i % 10 == 0:
        print(f'Current progress: {i}/{iterations}, Current approximation ratio: {-1*loss.item()/clEnergy}')
print(f'\nFinal Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

# ------------- Train QAOA without NN ------------- 

iterations = 60
QAOA_OptimizationWithoutNN(gammas,betas,iterations,costHamiltonianCircuit,optCirc,G,cost_h,clEnergy,2*configs-1)