# **Milestone P3: Team KLR**

## **1) Title** : Extending the comparison of classification models for rare civil war onsets


## **2) Abstract**

In similar fashion as Muchlinski et al., 2016, our study aims to contribute to the insofar discarded predictive statistical methods in political science, in favor of accurately predicting significant events such as civil wars. To this end, Muchlinski et al., 2016's approach of comparing experimental performance of algorithmic maneuveurs via Roc-Auc is adopted. However, in contrast to only using predictive algorithms, we attempt to couple a variety of classification algorithms with feature selection methods, which benefit an extended, deepened discussion surrounding this imbalanced prediction task. While indeed from Muchlinski et al., 2016, a clear winner prevails in all metrics employed, nuance of its strengths and weaknesses when paired with feature selection methods rather than arbitrary selection, and competing with better-suited methods, remains uninvestigated, providing justification for extending this analysis. 

## **3) Research Questions**

- Do different models determine different variables to be most predictive? (Consolidate previous findings)
- Do these findings agree with the features commonly believed to be most influential in predicting a Civil War that are discussed in Muchlinski et al., 2016? If not, why? (Enhance understanding of Civil War Onsets, and of statistical methods) 
- How do better suited models handle this class imbalance? (Improving prediction ability of these destructive events and the understanding of specific statistical methods)

## **4) Proposed dataset**

The dataset employed remains, for comparative purposes, the Civil War Data employed in Muchlinski et al., 2016. Cleanliness of the provided dataset is certainly a benefit and factor in being able to perform this wide-varied analysis given the short time-frame. Further, as different algorithms' performance may benefit or suffer from different dimensionality, varied feature selection will take place.  

## **5) Methods**

Annually measured Civil War Data, with {0,1} labels on the dependent dummy variable, indicating if a civil war outbreak took place, is provided by Hegre and Sambanis, 2006. Therein, is provided 7141 observations, with eighty-eight predictor variables. Online and offline feature selection methods are combined with Boosted trees, Support Vector Machines and Neural networks in order to compare these feature selection methods, and to leverage full ability from these models to handle larger or smaller feature dimensionalities in order to compare and discuss their predictive strengths and shortcomings. 

## **6) Proposed timeline**

The project is due for December 18th, giving us 3 weeks to complete the work. This is why we decided to split the work into 3 milestones:

- **December 7th**: Implement the statistical methods' computations and report graphs / results 
- **December 11th**: Clean up the code
- **December 18th**: Prepare the report and the short movie presentation

## **7) Organization within the team**

- **December 7th**: Implementation of the statistical methods' (SVM: Kamran, NN: Razvan, Boosted Trees: Loic). Feature selection will be implemented together and combined with the models once the models are set up to handle it, again each responsible for its incorporation into their method. Preparation of report graphs / results - each responsible for curating data from their statistical method in suitable format for the combined graphs (Roc-Auc curve, and F1 scores as in the original paper to preserve comparability. Incorporation of the previous methods too (Kamran) in order to also add comparability with our approach of incorporating feature selection)
- **December 11th**: Clean up the code (Razvan mostly)
- **December 18th**: Report and presentation, incorporating the previously prepared results (Loic and Kamran mostly)

## **8) Questions for TAs (optional)**

- Should we use other metrics than the ones from the original paper?
- Are the 3 methods that we intend to implement too many or too few for the scope of this project?
- Is AdaBoost a better choice for boosted trees since it is the most popular?
