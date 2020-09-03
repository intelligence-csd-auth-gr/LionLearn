# LionForests
<h4>LionForests (Local Interpretation Of raNdom FORESTS)</h4> 
Building interpretable random forests!

Towards a future where ML systems will integrate into every aspect of people’s lives, researching methods to interpret such systems is necessary, instead of focusing exclusively on enhancing their performance. Enriching the trust between these systems and people will accelerate this integration process. Many medical and retail banking/finance applications use state-of-the-art ML techniques to predict certain aspects of new instances. Thus, explainability is a key requirement for human-centred AI approaches. Tree ensembles, like random forests, are widely acceptable solutions on these tasks, while at the same time they are avoided due to their black-box uninterpretable nature, creating an unreasonable paradox. In this paper, we provide a methodology for shedding light on the predictions of the misjudged family of tree ensemble algorithms. Using classic unsupervised learning techniques and an enhanced similarity metric, to wander among transparent trees inside a forest following breadcrumbs, the interpretable essence of tree ensembles arises. An interpretation provided by these systems using our approach, which we call “LionForests”, can be a simple, comprehensive rule.

## Experiments
Dataset | No of Features | No of Instances | Type 
--- | --- | --- | ---
[Banknote](https://github.com/iamollas/LionLearn/blob/master/LionForests/LionForestsBanknoteExperiments.ipynb) | 4 | 1372 | Binary Classification
[Heart (Statlog)](https://github.com/iamollas/LionLearn/blob/master/LionForests/LionForestsHeartStatlogExperiments.ipynb) | 13 | 270 | Binary Classification
[Adult Census](https://github.com/iamollas/LionLearn/blob/master/LionForests/LionForestsAdultCencusExperiments.ipynb) | 14->80 | 48842 | Binary Classification

## Citation
Please cite the paper if you use it in your work or experiments :D :

- http://ceur-ws.org/Vol-2659/mollas.pdf
- https://arxiv.org/abs/1911.08780
- Accepted to NeHuAI of ECAI2020
