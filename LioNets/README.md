# LioNets 
LioNets (Local Interpretations Of Neural Networks thourgh Penultimate Layer Decoding) 

Building interpretable neural networks!

Technological breakthroughs on smart homes, self-driving cars, health care and robotic assistants, in addition to reinforced law regulations, have critically influenced academic research on explainable machine learning. A sufficient number of researchers have implemented ways to explain indifferently any black box model for classification tasks. A drawback of building agnostic explanators is that the neighbourhood generation process is universal and consequently does not guarantee true adjacency between the generated neighbours and the instance. This paper explores a methodology on providing explanations for a neural network's decisions, in a local scope, through a process that actively takes into consideration the neural network's architecture on creating an instance's neighbourhood, that assures the adjacency among the generated neighbours and the instance.


<table align="center">
    <tr>
        <td width="70%"> <img src="https://github.com/iamollas/LioNets/raw/master/lionetsArchitecture.png" width="100%"></td>
        <td width="30%"><p>In this picture is presenting the LioNets architecture. The main difference between LIME and LioNets is that the neighbourhood generation process takes place in the penultimate level of the neural network, instead of the original space. Thus, it is guaranteed that the generated neighbours will have true adjacency with the original instance, that we want to explain.</p></td>
    </tr>
</table>

## Citation
Please cite the paper if you use it in your work or experiments :D :

- https://arxiv.org/abs/1906.06566
- proceedings coming soon
