# LionForests
<h4>Conclusive Local Interpretation Rules for Random Forests through LionForests</h4> 

Due to their exceptionally high ability for analyzing vast quantities of data and making accurate decisions, machine learning systems are integrated into the health, industry and banking sectors, among others. On the other hand, their obscure decision-making processes lead to socio-ethical issues as they interfere with people's lives.  

In critical situations involving discrimination, gender inequality, economic damage, and even the possibility of casualties, machine learning models must be able to provide clear interpretations for their decisions. Otherwise, their obscure decision-making processes can lead to socioethical issues as they interfere with people's lives. In the aforementioned sectors, random forest algorithms strive, thus their ability to explain themselves is an obvious requirement. In this paper, we present LionForests, which relies on a preliminary work of ours. LionForests is a random forest-specific interpretation technique, which provides rules as explanations. It is applicable from binary classification tasks to multi-class classification and regression tasks, and it is supported by a stable theoretical background. Experimentation, including sensitivity analysis and comparison with state-of-the-art techniques, is also performed to demonstrate the efficacy of our contribution. Finally, we highlight a property of LionForest that distinguishes it from other techniques that do not have this property.

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
pickle.dumb(lf, open('lf_model.sav','w'))
...

#Load the LF instance
lf = pickle.load(open('lf_model.sav','rb'))

#Ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## Experiments
Type | # of  Datasets | Task 
--- | --- | --- 
Sensitivity Analysis | 3 | Binary Classification
Sensitivity Analysis | 3 | Multi-class Classification
Sensitivity Analysis | 3 | Regression
Time Analysis | 2 | Binary Classification
Comparison | 9 | All
Conclusive Check | 1 | Binary Classification
Qualitative Example | 1 | Binary Classification
Categorical Features | 1 | Binary Classification


## Citation
Please cite the paper if you use it in your work or experiments :D :

- LionForests complete version submitted to journal
- http://ceur-ws.org/Vol-2659/mollas.pdf
- https://arxiv.org/abs/1911.08780
- Accepted to NeHuAI of ECAI2020

## Contributors on Altruist
Name | Email
--- | ---
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr
[Grigorios Tsoumakas](https://intelligence.csd.auth.gr/people/tsoumakas/) | greg@csd.auth.gr
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr

## See our Work
[LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LioNets](https://github.com/iamollas/LionLearn/tree/master/LioNets): Local Interpretation Of Neural nETworkS through penultimate layer decoding
2. [Altruist](https://github.com/iamollas/Altruist): Argumentative Explanations through Local Interpretations of Predictive Models
3. [VisioRed](https://github.com/intelligence-csd-auth-gr/Interpretable-Predictive-Maintenance/tree/master/VisioRed%20Demo): Interactive UI Tool for Interpretable Time Series Forecasting called VisioRed