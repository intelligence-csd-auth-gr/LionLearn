# LioNets 
LioNets (Local Interpretations Of Neural Networks thourgh Penultimate Layer Decoding) 

Building interpretable neural networks!


![LioNets Architecture](https://github.com/iamollas/LioNets/raw/master/lionetsArchitecture.png)
In the above picture is presenting the LioNets architecture. The main difference between LIME and LioNets is that the neighbourhood generation process takes place in the penultimate level of the neural network, instead of the original space. Thus, it is guaranteed that the generated neighbours will have true adjacency with the original instance, that we want to explain.
