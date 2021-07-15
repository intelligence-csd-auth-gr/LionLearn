# LioNets (v2): A Neural-Specific Local Interpretation Technique Exploiting Penultimate Layer Information
<h4>LioNets (Local Interpretations Of Neural Networks thourgh Penultimate Layer Decoding)</h4> 

Building interpretable neural networks!

Artificial Intelligence (AI) has a tremendous impact on the unexpected growth of technology in almost every aspect. AI-powered systems are monitoring and deciding about sensitive economic and societal issues. The future is towards automation, and it must not be prevented. However, this is a conflicting viewpoint for a lot of people, due to the fear of uncontrollable AI systems. This concern could be reasonable if it was originating from considerations associated with social issues, like gender-biased, or obscure decision-making systems. Explainable AI (XAI) is recently treated as a huge step towards reliable systems, enhancing the trust of people to AI. Interpretable machine learning (IML), a subfield of XAI, is also an urgent topic of research. This paper presents a small but significant contribution to the IML community, focusing on a local-based, neural-specific interpretation process applied to textual and time-series data. The proposed methodology introduces new approaches to the presentation of feature importance based interpretations, as well as the production of counterfactual words on textual datasets. Eventually, an improved evaluation metric is introduced for the assessment of interpretation techniques, which supports an extensive set of qualitative and quantitative experiments.

## Experiments

This version is applied on multiple test cases containing:
| Dataset                                        	| Input                           	| Task                                 	| Source                      	|
|------------------------------------------------	|---------------------------------	|--------------------------------------	|-----------------------------	|
| SMS Spam Detection                             	| TFIDF Vectors (1000)            	| Binary Classification                	| https://bit.ly/3kNN65M      	|
| Ethos Binary: Hate Speech                      	| Word Embeddings (50x500)        	| Binary Classification                	| https://bit.ly/35MnUrL      	|
| Turbofan Engine Degradation Simulation Dataset 	| Matrix (Sliding window) (14x50) 	| Binary Classification and Regression 	| https://go.nasa.gov/3pHuJ6d 	| 

## Instructions
Please ensure you have docker installed on your desktop. Then:
```bash
docker pull johnmollas/lionets
```
After succesfully installing LioNets, please do:
```bash
docker run -p 8888:8888 johnmollas/lionets
```
Then, in your terminal copy the localhost url and open it in your browser. Enjoy :)

If you want to build the docker by yourself please zip the folder lionets with the requirements.txt file and name the zip "lionets.zip". Then:
```bash
docker build -t my_lionets_docker .
```
After succesfully building LioNets, please do:
```bash
docker run -p 8888:8888 my_lionets_docker
```

## Citation *Soon to be available*
Please cite the paper if you use it in your work or experiments :D :

- https://arxiv.org/abs/2104.06057

## Contributors on Altruist
Name | Email
--- | ---
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr
[Grigorios Tsoumakas](https://intelligence.csd.auth.gr/people/tsoumakas/) | greg@csd.auth.gr
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
