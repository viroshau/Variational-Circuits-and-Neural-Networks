# Variational Circuits and Neural Networks
 Files for my master thesis on the use of machine learning in variational quantum circuits

## Video of the basic toy landscape where the first jump fails to reach the global minimum, but second jump succeeds (Nelder mead)

First jump:
https://user-images.githubusercontent.com/51076750/174819631-6af34bbf-3e1b-4305-83d1-8670a1df11ea.mp4

Second jump:
https://user-images.githubusercontent.com/51076750/174819692-94cd2ba1-88bb-44ea-8771-63d94cb52c76.mp4

## Video of the toy landscape with setup 1 where the hidden layer has ReLU as activation function

First jump:

Second jump:

## Video of the ESCAPE procedure on the QAOA landscape
The following video illustrates the ESCAPE procedure using $g(t) = t/T$ at step 4 of the procedure. As the video illustrates, the procedure actively alters the QAOA landscape in a way to escape the initially found local minima. The initial frame is in the completely altered landcape, and the landscape at various $t$ are animated. The SPSA minimzer is used for 5 steps at each $t$ to approximate a single gradient step. The green line corresponds to the initial minimization that gets stuck in a subpar local minima, while the yellow line corresponds to the trajectory taken at step 4 of the procedure.

https://user-images.githubusercontent.com/51076750/174802234-605c2a70-9feb-4e48-bb44-5f2ac23bd465.mp4

