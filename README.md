Diabetes Prediction using R

The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome ( whether the person has diabetes (1 = Yes, 0 = No)) Independent variables include;
•	Pregnancies – number of times pregnant
•	Glucose – blood sugar level
•	BloodPressure – blood pressure value
•	SkinThickness – skin fold thickness
•	Insulin – insulin level
•	BMI – body mass index
•	DiabetesPedigreeFunction – family history score
•	Age – patient’s age

Methods Used
1.   Logistic Regression
   - Stepwise regression used for variable selection
   - Removal of insignificant predictors 

2.   Decision Tree (rpart & ctree)
   - Visualized with `rpart.plot` and `party` packages
   - Provides interpretable classification rules

3.   Naïve Bayes
   - Applied with repeated train/test splits
   - Accuracy averaged across 10 experiments

4.   Model Evaluation
   - Confusion Matrix
   - Accuracy comparison

 Interpretation & Insights

 Data Exploration
•	Glucose, BMI, and Age showed distinct patterns between people with and without diabetes.
•	The t-tests confirmed that average glucose and Diabetes Pedigree Function differ significantly between the two groups.
•	Correlation analysis showed some predictors are related, but not strongly enough to cause redundancy.

Logistic Regression; A full logistic regression with all variables showed that:
•	Glucose, BMI, Pregnancies, and Diabetes Pedigree Function were strong predictors.
•	SkinThickness, BloodPressure, Insulin, and Age were not statistically significant (p-value > 0.01).
•	Stepwise regression confirmed that removing the less significant predictors still maintained predictive power.
•	Glucose level is the single most important factor for predicting diabetes, followed by BMI and pregnancy history.

Outlier Detection
•	Local Outlier Factor (LOF) identified a few unusual patient records.
•	Outliers may represent measurement errors or special cases, but they did not significantly distort the overall analysis.

Decision Trees
•	The first decision tree split mainly on Glucose and BMI, reinforcing their predictive strength.
•	After pruning (using the complexity parameter), the tree became simpler and more interpretable.

Conditional Inference Tree (CTree)
•	Produced similar results to the rpart tree, with slightly different splits.
•	Confusion matrix showed decent classification accuracy.

 Naïve Bayes
•	Repeated experiments (10 runs with random splits) showed stable accuracy.
•	Average accuracy was reasonable but slightly lower than logistic regression and trees.

Conclusion
•	Glucose level and BMI are the most reliable predictors of diabetes.
•	Logistic regression is effective for identifying risk factors, while decision trees provide interpretable rules useful in practice.
•	Naïve Bayes performs adequately but may not capture variable interactions as well as regression or trees.
•	Outlier detection is useful for data cleaning but does not change the main findings.
•	Overall, Preventive efforts and clinical screening should focus on individuals with high glucose and high BMI, as these are the strongest indicators of diabetes risk.

Recommendations
•	Prioritize screening for individuals with high glucose and high BMI, since these were the strongest predictors of diabetes.
•	Incorporate simple decision tree rules into routine health checks e.g., if a patient’s glucose is above a threshold, flag them for further testing.
•	Use logistic regression results to identify patients at borderline risk and advise lifestyle changes early.
•	Apply advanced machine learning models (Random Forest, XGBoost, Neural Networks) and compare with the baseline models here.




