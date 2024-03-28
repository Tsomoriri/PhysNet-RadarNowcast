# PhysNet-RadarNowcast
Applying PINN to radar data for nowcasting weather

redo input data and output data config
add cfl
add loss for mass conservation
maybe more resolution

_____________________________
meeting with stefan and IS (15/03/2024)

change in ux and uy --> trainable
explore:
 diffferent NAS parameters
 different loss functions
 different way to calculate accuracy
 boundary conditions

explore more equations simple
conv lstm radar


-------------------------------------
meeting with IS 26/03/2024

things to do:
- conv lstm read and understand how it works
- stefan paper read and understand
- explainable ai metrics- understand and add to my demo
- add a comparison of how values change in physics parameters vs the neural network parameters
- ablation graph of physics loss and neural network loss
- write up the research proposal
- read on how physics loss is implemented pinn read up 
- understand how to add physics loss, is it same dimension or another
- understand how diferent people are doing pinns in real world.- nvidia modulus source code.
- research paper acm style summary- of what i need to do
- background reading on how PINNs are used in weather forecasting

-----------------------------------------------------------
meeting with stefan 28/03/2024

things to do:
- 
