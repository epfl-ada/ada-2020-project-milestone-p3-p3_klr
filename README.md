# **Milestone P3: Team KLR**

## **1) Title** : Extending the comparison of classification algorithms' ability to predict imbalanced civil war data


## **2) Abstract**

In similar fashion as Muchlinski et al., 2016, this paper aims to contribute to the insofar discarded predictive statistical methods in political science, in favor of accurately predicting significant events such as civil wars. To this end, Muchlinski et al., 2016's approach of comparing experimental performance of algorithmic maneuveurs via a multitude of metrics is adopted. In contrast, however, attempts to reach beyond the inclusion of only Random Forests take place, with rigorous incorporation of stacking, boosting, and a variety of classification algorithms, which benefit an extended, deepened comparison and discussion surrounding not only this imbalanced prediction task, but also the causal estimation of features. While indeed from Muchlinski et al., 2016, it seems that Random Forests drastically outperform their determined competitor in all metrics employed, nuance of their strengths and weaknesses in contrast with more suited models remains uninvestigated, providing justification for extending this analysis. 

## **3) Research Questions**

- Do different models determine different variables to be most predictive? (Consolidate previous findings)
- Do these findings agree with the features commonly believed to be most influential in predicting a Civil War that are discussed in Muchlinski et al., 2016? If not, why? (Enhance understanding of Civil War Onsets, and of statistical methods) 
- How do better suited models handle this class imbalance? (Improving prediction ability of these destructive events and the understanding of specific statistical methods)

## **4) Proposed dataset**

The dataset employed remains, for comparative purposes, the Civil War Data employed in Muchlinski et al., 2016. Cleanliness of the provided dataset is certainly a benefit and factor in being able to perform this wide-varied analysis given the short time-frame. Further, as different algorithms' performance may benefit or suffer from different dimensionality, varied feature selection will take place.  

## **5) Methods**

Annually measured Civil War Data, with labels on the dependent variable, $Y_{cw}^{ij}$, which is a dummy variable of
whether a civil war onset occurred for a given country, i, in a given year j, is provided by Hegre and Sambanis, 2006. Therein, is provided 7141 observations, with eighty-eight predictor variables. ...

## **6) Proposed timeline**

The project is due for December 18th, giving us 3 weeks to complete the work. This is why we decided to split the work into 3 milestones:

- **December 7th**: Implement the statistical methods' computations and report graphs / results 
- **December 11th**: Clean up the code
- **December 18th**: Prepare the report and the short presentation movie

## **7) Questions for TAs (optional)**

- Libraries to use?
- Heuristics?
