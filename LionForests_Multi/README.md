# LionForests
<h4>Local Multi-Label Explanations for Random Forest</h4> 

Due to their exceptionally high ability for analyzing vast quantities of data and making accurate decisions, machine learning systems are integrated into the health, industry and banking sectors, among others. On the other hand, their obscure decision-making processes lead to socio-ethical issues as they interfere with people's lives.  

![LionForests Flowchart](https://github.com/intelligence-csd-auth-gr/LionLearn/blob/master/LionForests/LionForestsFlow.png?raw=true)

Multi-label classification is a challenging task, particularly in domains where the number of labels to be predicted is large. Deep neural networks are often effective at multi-label classification of images and textual data. When dealing with tabular data, however, conventional machine learning algorithms, such as tree ensembles, appear to outperform competition. Random forest, being the most common of these ensembles, has found use in a wide range of real-world problems. Such problems include fraud detection in the financial domain, crime hotspot detection in the legal sector, and in the biomedical field, disease probability prediction when patient records are accessible. Since they have an impact on people's lives, these domains usually require decision-making systems to be explainable. Random Forest falls short on this property, especially when a large number of tree predictors are used. This issue was addressed in a recent research named LionForests, regarding single label classification and regression. In this work, we adapt this technique to multi-label classification problems, by employing three different strategies regarding the labels that the explanation covers. Finally, we provide a set of qualitative and quantitative experiments to assess the efficacy of this approach.

## Run through Docker
Clone this repository and navigate to the folder LionForests/Docker. There run these commands
```bash
docker build -t lionforests-multi .
docker run -p 8888:8888 lionforests-multi
```
Then, just open the url the terminal prined and you are ready to play with the notebooks. 

## Pull from Docker and pip install
```bash
docker pull johnmollas/multi-lf .
docker run -p 8888:8888 johnmollas/multi-lf
```

## Requirements
For the requirements just check the requirements.txt file. LF in order to run properly needs these libraries. The libraries anchor-exp, dask-ml, imbalanced-learn and pyfpgrowth, are necessary for the rest of the algorithms we used in our comparisons (CHIRPS, Anchors, MARLENA).


## Example #1
```python
X, y, feature_names, class_names = load_your_data()
lf = LionForests(None, False, None, feature_names, class_names) #first none means that no RF model is provided, second none means no scaling
lf.fit(X, y) #will grid search to find the best RF for your data

#ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## Example #2
```python
X, y, feature_names, class_names = load_your_data()
lf = LionForests(rf_model, True, None, feature_names, class_names) #now we provide a model
lf.fit_trained(X, y) #however, LF needs few statistics to be extracted from training data

#ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## How to save and reuse
```python
#Use one of the above examples to build your LF instance
... lf

import pickle
#Save the whole LF instance, which contains the model and the data statistics (but not the data themselves)
pickle.dump(lf, open('lf_model.sav','wb'))
...

#Load the LF instance
lf = pickle.load(open('lf_model.sav','rb'))

#Ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## Experiments
Type | # of  Datasets | Task 
--- | --- | --- 
Comparison | 4 | Multi-Label Classification
Qualitative Example | 1 | Multi-Label Classification


## Citation
Please cite the paper if you use it in your work or experiments :D :
- [Journal] :
    - LionForests complete version published at [DAMI]: https://link.springer.com/article/10.1007/s10618-022-00839-y
    - https://arxiv.org/abs/2104.06040, available on Arxiv as well
- [Conference] 
    - To Appear in Workshop Proceedings

## Contributors on Multi LionForests
Name | Email
--- | ---
[Nikolaos Mylonas](https://intelligence.csd.auth.gr/people/people-nikos-mylonas-phd-student/) | myloniko@csd.auth.gr
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr
[Grigorios Tsoumakas](https://intelligence.csd.auth.gr/people/tsoumakas/) | greg@csd.auth.gr
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr

## See our Work
[LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LioNets](https://github.com/iamollas/LionLearn/tree/master/LioNets): Local Interpretation Of Neural nETworkS through penultimate layer decoding
2. [Altruist](https://github.com/iamollas/Altruist): Argumentative Explanations through Local Interpretations of Predictive Models
3. [VisioRed](https://github.com/intelligence-csd-auth-gr/Interpretable-Predictive-Maintenance/tree/master/VisioRed%20Demo): Interactive UI Tool for Interpretable Time Series Forecasting called VisioRed
