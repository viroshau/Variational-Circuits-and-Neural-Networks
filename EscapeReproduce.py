from QuantumCircuits import *


def customcost(gammas,betas,G,probcircuit,neuralNet,adjacencymatrix,configurations):
    probs = probcircuit(gammas,betas,G) #size = (2^{len(G.nodes)})
    x = torch.sign(neuralNet(configurations)) #The predictions from the neural network based on the output. Note that configurations is a list with all possible bitstring outcomes from the quantum system
    energiesOfConfigurations = EvaluateCutOnDataset(x,adjacencymatrix)*probs #weighs the energies of each configurations with their probabilities
    return torch.sum(energiesOfConfigurations) #Returns the weighted sum of energies using the probabilities of obtaining each output string

foldername = './ESCAPEResults'
#Define graph and and cost Hamiltonian
G = CreateRegularGraph(5,4,True)
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,left,right = BestClassicalHeuristicResult(G)
cost_h = createCostHamiltonian(G,adjacencymatrix) #cost_h, mixer_h = qml.qaoa.maxcut(G)

configs = torch.tensor(CreateBinaryList(len(G.nodes)),dtype = torch.float32)

#Define Devices to run the circuits
NUMSHOTS = 500
devExact = qml.device('default.qubit.torch', wires = len(G.nodes)) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
devShot = qml.device('default.qubit.torch',wires = len(G.nodes),shots = NUMSHOTS)

costHamiltonianCircuit = qml.QNode(QAOAreturnCostHamiltonian,devExact,interface = "torch",diff_method = 'best') #Returns the cost value after running the circuit
probabilityCircuit = qml.QNode(QAOAreturnProbs,devExact,interface = "torch") #Returns the state probability before measurement
shotCircuit = qml.QNode(QAOAreturnSamples,devShot,interface = 'torch') #Returns a set of measurement samples

#Turn gammas and betas into trainable variables
p = 8
simulations = 100
gammaslist = torch.tensor(np.random.uniform(low = 0, high = 2*np.pi, size = (simulations,p)))
betaslist = torch.tensor(np.random.uniform(low = 0, high = 2*np.pi,size = (simulations,p)))

QAOAIterations = 60
QAOAlr = 0.3

hybridQAOANNSteps = 350

initialEnergies = np.zeros(simulations)
finalEnergies = np.zeros(simulations)
for j in tqdm(range(simulations)):

    gammas = torch.autograd.Variable(gammaslist[j],requires_grad = True)
    betas = torch.autograd.Variable(betaslist[j],requires_grad = True)
    # ------------- Initial Optimization of VQC parameters ------------- 
    optQAOA = torch.optim.Adam([gammas,betas],lr = QAOAlr)

    VQCOptimizationlosses = QAOA_OptimizationWithoutNN(gammas,betas,QAOAIterations,costHamiltonianCircuit,optQAOA,G,cost_h,clEnergy,2*configs-1)
    initialEnergies[j] = VQCOptimizationlosses[-1]

    # ------------- Train the NN using VQC circuit as sample generator ------------- 

    #Sample form the circuit to generate the strings that the NN should be trained with

    #Train the Neural Network
    model = OneLayerNN(len(G.nodes),len(G.nodes))

    tot_epoch = 80
    iterationsNN = 1
    NN_lr = 0.05
    opt = torch.optim.SGD(model.parameters(),lr = NN_lr)
    
    losses = NN_Optimization(gammas,betas,tot_epoch,iterationsNN,G,NUMSHOTS,model,shotCircuit,adjacencymatrix,clEnergy,opt)

    # ------------- Train QAOA using varying NN parameters ------------- 

    #Perform steps for optimization

    initialweights = model.return_weights() #Used when the NN landscape is gradually changed.

    #print('Training VQC using a changing Neural Network: \n ------------------------------------- \n')
    #print(f'Initial Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

    results = np.zeros(hybridQAOANNSteps)
    for i in range(hybridQAOANNSteps):
        #Change the weights of the Neural Network

        weights = changingNNWeights(initialweights,hybridQAOANNSteps,i,g_heaviside)
        model.set_custom_weights(model,weights)

        #Perform optimization steps
        optQAOA.zero_grad()
        loss = customcost(gammas,betas,G,probabilityCircuit,model,adjacencymatrix,2*configs-1)
        loss.backward()
        optQAOA.step()

        results[i] = loss.item()
        #Print status of the simulation
        #if i % 10 == 0:
        #    print(f'Current progress: {i}/{iterations}, Current Energy: {1*loss.item()}')
    #print(f'Final Parameters: \nGamma = {gammas.data.numpy()} \nBeta = {betas.data.numpy()}')

    # ------------- Final training QAOA without NN ------------- 

    VQCOptimizationlosses2 = QAOA_OptimizationWithoutNN(gammas,betas,QAOAIterations,costHamiltonianCircuit,optQAOA,G,cost_h,clEnergy,2*configs-1)
    
    finalEnergies[j] = VQCOptimizationlosses2[-1]

    fig, axs = plt.subplots(2, 2)    
    fig.suptitle(f'Best classical energy: {clEnergy}')

    axs[0,0].plot(list(range(QAOAIterations)),VQCOptimizationlosses)
    axs[0,0].set_title('Initial QAOA optimization')
    axs[0,0].set_xlabel('Iterations')
    axs[0,0].set_ylabel('loss')

    axs[0,1].plot(list(range(tot_epoch)),losses)
    axs[0,1].set_title('Neural network optimization')
    axs[0,1].set_xlabel('Iterations')
    axs[0,1].set_ylabel('loss')

    axs[1,0].plot(list(range(hybridQAOANNSteps)),results)
    axs[1,0].set_title('QAOA optimization with varying NN')
    axs[1,0].set_xlabel('Iterations')
    axs[1,0].set_ylabel('loss')

    axs[1,1].plot(list(range(QAOAIterations)),VQCOptimizationlosses2)
    axs[1,1].set_title('Final QAOA optimization')
    axs[1,1].set_xlabel('Iterations')
    axs[1,1].set_ylabel('loss')
    plt.show()
    #QAOA_OptimizationWithoutNN(gammas,betas,iterations,qcircuit,opt,G,cost_h,clEnergy)

print(initialEnergies)
print(finalEnergies)

energies = np.zeros((2,len(initialEnergies)))
energies[0] = initialEnergies
energies[1] = finalEnergies
savestring = f'/ADAM,p={p},SGD_NN_lr={NN_lr},Adam_VQC_lr={0.3},M={tot_epoch},T={350},QAOA_iter = {QAOAIterations},1.py'

np.save(foldername+savestring,energies)

fig, axs = plt.subplots(2, sharex=True)
axs[0].hist(list(range(2**len(G.nodes))),weights = probabilityCircuit(gammas,betas,G).detach().numpy(),bins = 2**len(G.nodes),label = 'Probs from algo',color = 'y')
axs[1].hist(list(range(2**len(G.nodes))),weights = -1*EvaluateCutOnDataset(2*configs-1,adjacencymatrix).detach().numpy(),bins = 2**len(G.nodes),label = 'Energy distribution')
axs[1].set(xlabel = 'Bitstrings')
fig.legend(loc = 'center',ncol = 2)
plt.show()


#Coverance matrix (will blow up NN) (maybe don't do it)
#The more negative the loss, the better the solution