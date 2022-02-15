import torch
from torch.autograd import Function
import torch.optim as optim
from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister,execute
from qiskit.circuit import Parameter
from qiskit import Aer
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.autograd import Function

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list:
        num_list += [tensor.item()]
    return num_list

class QiskitCircuit_QAOA():
    def __init__(self,shots):
        self.beta = Parameter('Beta')
        self.gamma = Parameter('Gamma')
        self.shots = shots
        
        def create_circuit():
            ckt = QuantumCircuit(2, 2)
            # add mixer part
            ckt.rx(self.beta, 0)
            ckt.rx(self.beta, 1)

            # add H_target part, for each Zi Zj do this
            ckt.cx(0,1)
            ckt.rz(-1*self.gamma, 1)
            ckt.cx(0,1)
            ckt.measure([0,1],[0,1])
            return ckt

        self.circuit = create_circuit()
    
    def energy_expectation(self, counts, shots, i,j, Cij=-1): #calculate expectation for one qubit pair
        expects = 0
        #print(counts)
        for key in counts.keys():
            perc = counts[key]/shots
            check = Cij*(float(key[i])-1/2)*(float(key[j])-1/2)*perc
            expects += check
        return [expects] 
    
    def bind(self,parameters):
        [self.beta,self.gamma] = parameters
        #print(self.circuit.data)
        self.circuit.data[0][0]._params = to_numbers(parameters)[0:1]
        self.circuit.data[1][0]._params = to_numbers(parameters)[0:1]
        self.circuit.data[3][0]._params = to_numbers(parameters)[1:2]
        return self.circuit
 
    def run(self, i):
        self.bind(i)
        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(self.circuit,backend,shots=self.shots)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        #print(counts)
        return self.energy_expectation(counts, self.shots, 0,1)    
    
class TorchCircuit(Function):    

    @staticmethod
    def forward(ctx, i):
        if not hasattr(ctx, 'QiskitCirc'):
            ctx.QiskitCirc = QiskitCircuit_QAOA(shots=10000)
            
        #print(i[0])    
        #print(i)
        #print(i[0])
        exp_value = ctx.QiskitCirc.run(i[0])
        
        result = torch.tensor([exp_value])
        
        ctx.save_for_backward(result, i) #Save the result and parameters for backwards
        print('re:', result)
        return result #only return the expectation value for forward pass
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = 0.01
        
        forward_tensor, i = ctx.saved_tensors #Saved the  forward call for the sake of finite diff, parameters needed too. It's a list in a list
        input_numbers = to_numbers(i[0]) #Turn parameters into python list for qiskit
        #print(forward_tensor)
        gradient = [] #Store gradient values
        
        for k in range(len(input_numbers)): #For k in all inputs
            input_eps = input_numbers 
            input_eps[k] = input_numbers[k] + eps #input[k] += eps because of parameter shift

            exp_value = ctx.QiskitCirc.run(torch.tensor(input_eps))[0] #Run the shifted circuit
            result_eps = torch.tensor([exp_value])
            gradient_result = (exp_value - forward_tensor[0][0].item())/eps #Calculat the finite diff gradient. Note forward_tensor[0][0].item()
            gradient.append(gradient_result)
            
        print('gradient= ',gradient)
        result = torch.tensor([gradient])
        print('res:',result)
#         print(result)
        #print("test tens size", result.float(), grad_output.float())
        print(result.float() * grad_output.float())
        return result.float() * grad_output.float() #Apply grad_output onto both gradients calculated by finite difference

torch.manual_seed(42)  
x = torch.tensor([[0.3, 0.2]], requires_grad=True)

qc = TorchCircuit.apply
y1 = qc(x)

y1.backward()
stop
print(x.grad)

qc = TorchCircuit.apply

def cost(x):
    target = -0.25
    expval = qc(x)
    return torch.abs(qc(x).sum() - target) ** 2, expval

x = torch.tensor([[np.pi/4, np.pi/4]], requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)

num_epoch = 100

loss_list = []
expval_list = []

for i in tqdm(range(num_epoch)):
# for i in range(num_epoch):
    opt.zero_grad()
    loss, expval = cost(x)
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    expval_list.append(expval.item())
#     print(loss.item())

plt.plot(loss_list)
plt.show()