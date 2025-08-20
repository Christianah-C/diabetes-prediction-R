diabetes <- read.csv("C:\\Users\\Christianah.O_BROOKS\\Downloads\\diabetes.csv", header=T, stringsAsFactors = F)
summary(diabetes)

par(mfrow =c(2,2))
hist(diabetes$Pregnancies)
hist(diabetes$Age)
hist(diabetes$Glucose)
hist(diabetes$BMI)
boxplot(diabetes$BloodPressure, ylab = "BloodPressure")
par(mfrow =c(1,1))

install.packages(ggplot)
library(ggplot2)
ggplot(diabetes,aes(x=Glucose))+geom_histogram(fill = "skyblue", colour = "black")+ facet_grid(Outcome~.)

ggplot(diabetes,aes(x=Glucose))+geom_histogram(fill = "skyblue", colour = "black")+ facet_grid(.~Outcome)

t.test(Glucose ~ Outcome, diabetes)

par(mfrow = c(1,2))

#boxplot

with(diabetes, boxplot(DiabetesPedigreeFunction ~ Outcome,
           ylab = "DiabetesPedigreeFunction(DPF)",
           xlab = "Diabetes Presence",
           main = "Plot 1",
           outline = TRUE))

with_d <- diabetes [diabetes$Outcome == 1,]
without <- diabetes [diabetes$Outcome == 0, ]

#density plot
plot(density(with_d$Glucose),
     xlim = c(0, 250),
     ylim = c(0.00, 0.02),
     xlab = "Glucose level",
     main = "Plot 2",
     lwd = 2)
lines(density(without$Glucose),
      col = "orange",
      lwd = 2)
legend("topleft",
       col = c("black", "orange"),
      legend = c("With Diabetes", "Without Diabetes"),
      lwd = 2,
      bty = "n")

#two sample ttest with unequal variance
t.test(with_d$DiabetesPedigreeFunction, without$DiabetesPedigreeFunction)

install.packages("GGally")   # Run this only once
library(GGally)              # Load the package every session

#Correlation between each variable
#scatter matrix of all columns

ggcorr(diabetes[,-9], name = "corr", label = TRUE)+
  theme(legend.position = "none")+
  labs(title = "Correlation of Variance")+
  theme(plot.title=element_text(face='bold',color='black', hjust =0.5, size

#Fitting a logistic regression to assess importance of predictors
method <- paste0(paste(names(diabetes)[length(diabetes)], collapse ="+"))
logistic <- glm(Outcome ~ ., family = binomial, data = diabetes)
logistic
summary(logistic)

#Features Selection

Model_coeff <- exp(coef(logistic))[2:ncol(diabetes)]
Model_coeff <- Model_coeff[c(order(Model_coeff,decreasing=TRUE)[1:(ncol(diabetes)-1)])]
predictors_names <- c(names(Model_coeff),names(diabetes),names(diabetes)[length(diabetes)])

predictors_names


#filter df with n most important predictors
diabetes_df <- diabetes[, c(predictors_names)]
head(diabetes_df)

#outlier detection
install.packages("dbscan")
library(dbscan)
outlier_scores <- lof(diabetes_df, minPts=5)
plot(density(outlier_scores))

outliers <- order(outlier_scores, decreasing=T) [1:5]
print(outliers)

n <- nrow(diabetes_df)
labels <- 1:n
labels [-outliers] <- "."
pc <- prcomp(outcome, scale. = TRUE)
biplot(pc, cex = 0.8, xlabs = labels)

biplot(prcomp(diabetes_df, cex=.8, xlabs = labels))

install.packages("Rlof")
library(Rlof)
outlier_scores <- lof(diabetes_df, k=5)
outlier_scores <- lof(diabetes_df, k=c(5:10))
outlier_scores

#1st Model
First_Model <- glm(formula=Outcome~., family = binomial, data=train)
summary(First_Model)


model <- glm(Outcome ~ ., data = diabetes, family = binomial)
smodel <- step(model) #stepwise logistic regression


install.packages("rpart")
library(rpart)
tree <- rpart(Outcome~., data=diabetes, method= "class")

install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(tree)

#complexity parameter
plotcp(tree)

tree1 <- rpart(Outcome~., data=diabetes, method ="class", cp=0.016)
rpart.plot(tree1)

#2nd Model

set.seed(123)  # for reproducibility

# Create index for 70% training data
index <- sample(seq_len(nrow(diabetes)), size = 0.7 * nrow(diabetes))

# Split into training and test sets
train <- diabetes[index, ]
test  <- diabetes[-index, ]

log_model <- glm(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness +
                   Insulin + BMI + DiabetesPedigreeFunction,
                 family = binomial, data = train)
summary(log_model)

par(mfrow = c(2,2))

plot(log_model)

#3rd Model Predict Diabetes Risk on new patients using Decision Tree

install.packages("partykit")

# Load the library
library(partykit)

ct <- ctree(Outcome ~ ., data = train)

prediction_probability <- predict(ct, test, type = c("prob"))
prediction_class <- predict(ct, test, type = c("response"))

table(prediction_class, test$Outcome)

library(caret)

# Make sure both are factors with same levels
prediction_class <- factor(prediction_class, levels = c(0, 1))
test$Outcome <- factor(test$Outcome, levels = c(0, 1))

# Confusion matrix
con_m <- confusionMatrix(prediction_class, test$Outcome,
                         positive = NULL,   # set positive class
                         dnn = c("Prediction", "Reference"))

con_m

#4th Model Naïve Bayes
Accuracy_p <- numeric(10)

for (l in 1:10) {
  sample_size <- floor(0.90 * nrow(diabetes))
  train_ind <- sample(seq_len(nrow(diabetes)), size = sample_size)

  train <- diabetes[train_ind, ]
  test  <- diabetes[-train_ind, ]

  # Convert Outcome to factor
  train$Outcome <- as.factor(train$Outcome)
  test$Outcome  <- as.factor(test$Outcome)

  install.packages("e1071")   # naivebayes
  library(e1071)

  # Train Naive Bayes
  nb <- naiveBayes(Outcome ~ ., data = train)

  # Predict on test set
  z  <- predict(nb, test)

  # Confusion matrix & accuracy
  Acc <- table(test$Outcome, z)
  Accuracy_p[l] <- sum(diag(Acc)) / sum(Acc) * 100
}

# Store experiment results
Experiments <- c(1:10)
NAIVE_Bayes <- data.frame(Experiments, Accuracy_p)
NAIVE_Bayes

# Average accuracy across runs
Average <- mean(Accuracy_p)
Average

}






