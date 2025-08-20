# Diabetes Prediction using R

This project applies **statistical models and machine learning techniques** to the **Pima Indians Diabetes dataset**.  
The goal is to predict whether a patient has diabetes (`Outcome`) based on medical diagnostic measurements.

---

## 📂 Project Structure
- `diabetes_analysis.R` → R script containing all model building and evaluation steps  
- `pima_diabetes.csv` → dataset (optional: add if publicly shareable)  
- `plots/` → generated plots (e.g., decision trees, ROC curves)  

---

## ⚙️ Methods Used
1. **Logistic Regression**
   - Stepwise regression used for variable selection
   - Removal of insignificant predictors (e.g., `SkinThickness`, `Insulin`, `Age`)

2. **Decision Tree (rpart & ctree)**
   - Visualized with `rpart.plot` and `party` packages
   - Provides interpretable classification rules

3. **Naive Bayes**
   - Applied with repeated train/test splits
   - Accuracy averaged across 10 experiments

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy comparison
   - Interpretation of significant predictors

---

## 📊 Key Results
- Logistic regression identified **Glucose, BMI, and Pregnancies** as strong predictors.  
- Decision trees provided interpretable rules (e.g., thresholds on glucose and BMI).  
- Naive Bayes achieved an **average accuracy of ~X%** (replace with your computed value).  
- Models were evaluated using confusion matrices and accuracy scores.

---

## 🚀 How to Run
Run the R script in RStudio or terminal:

```R
# install required packages
install.packages(c("caret", "MASS", "rpart", "rpart.plot", "e1071", "party"))

# run the analysis
source("diabetes_analysis.R")

