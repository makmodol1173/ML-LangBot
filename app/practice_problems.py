"""
Practice Problems Module
Auto-generates exercises with solutions for each ML topic
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split


class PracticeProblemsGenerator:
    """Generates practice problems and solutions for ML algorithms"""
    
    def __init__(self):
        self.problem_templates = self._initialize_problem_templates()
        self.coding_templates = self._initialize_coding_templates()
        self.difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
    
    def _initialize_problem_templates(self) -> Dict[str, Dict]:
        """Initialize comprehensive problem templates for all ML topics"""
        return {
            
            # Data Preprocessing
            "Missing Data": {
                "problems": [
                    {
                        "question": "You have a dataset with 20% missing values in the 'age' column. Which imputation strategy would be most appropriate?",
                        "options": ["Mean imputation", "Median imputation", "Mode imputation", "Forward fill"],
                        "correct": 1,
                        "explanation": "Median imputation is robust to outliers and works well for numerical data with missing values."
                    },
                    {
                        "question": "What's the best approach for handling missing categorical data?",
                        "options": ["Mean imputation", "Mode imputation", "Delete rows", "Create 'Unknown' category"],
                        "correct": 3,
                        "explanation": "Creating an 'Unknown' category preserves data and can provide meaningful insights."
                    },
                    {
                        "question": "When should you use listwise deletion for missing data?",
                        "options": ["Always", "When missing data is < 5%", "When data is MCAR", "Never"],
                        "correct": 2,
                        "explanation": "Listwise deletion is appropriate when data is Missing Completely At Random (MCAR) and the amount is small."
                    },
                    {
                        "question": "What is the main disadvantage of mean imputation?",
                        "options": ["Increases variance", "Reduces variance", "Changes the mean", "Adds bias"],
                        "correct": 1,
                        "explanation": "Mean imputation reduces variance by replacing missing values with the same value (mean)."
                    },
                    {
                        "question": "Which technique can handle missing data during model training?",
                        "options": ["Linear Regression", "Random Forest", "SVM", "K-Means"],
                        "correct": 1,
                        "explanation": "Random Forest can handle missing values internally by using surrogate splits."
                    }
                ]
            },
            
            "Encoding": {
                "problems": [
                    {
                        "question": "When should you use One-Hot Encoding vs Label Encoding?",
                        "options": ["Always use One-Hot", "One-Hot for nominal, Label for ordinal", "Always use Label", "They're the same"],
                        "correct": 1,
                        "explanation": "One-Hot Encoding for nominal categories (no order), Label Encoding for ordinal categories (with order)."
                    },
                    {
                        "question": "What problem does One-Hot Encoding solve?",
                        "options": ["Missing data", "Ordinal relationships", "Curse of dimensionality", "Artificial ordering"],
                        "correct": 3,
                        "explanation": "One-Hot Encoding prevents algorithms from assuming artificial ordering in categorical variables."
                    },
                    {
                        "question": "What is the curse of dimensionality in One-Hot Encoding?",
                        "options": ["Too few features", "Too many features", "Missing values", "Duplicate features"],
                        "correct": 1,
                        "explanation": "One-Hot Encoding can create too many features when dealing with high-cardinality categorical variables."
                    },
                    {
                        "question": "Which encoding technique is best for high-cardinality categorical variables?",
                        "options": ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Binary Encoding"],
                        "correct": 2,
                        "explanation": "Target Encoding uses the target variable to encode categories, reducing dimensionality for high-cardinality variables."
                    },
                    {
                        "question": "What is dummy variable trap?",
                        "options": ["Too many categories", "Perfect multicollinearity", "Missing categories", "Wrong encoding"],
                        "correct": 1,
                        "explanation": "Dummy variable trap occurs when dummy variables are perfectly correlated, causing multicollinearity."
                    }
                ]
            },
            
            "Feature Scaling": {
                "problems": [
                    {
                        "question": "When should you use StandardScaler vs MinMaxScaler?",
                        "options": ["Always use StandardScaler", "Use MinMaxScaler for neural networks", "Use StandardScaler when data is normally distributed", "They're interchangeable"],
                        "correct": 2,
                        "explanation": "StandardScaler works best with normally distributed data, while MinMaxScaler is better for bounded ranges."
                    },
                    {
                        "question": "What does StandardScaler do to your data?",
                        "options": ["Scales to 0-1 range", "Centers around 0 with unit variance", "Removes outliers", "Normalizes to unit vector"],
                        "correct": 1,
                        "explanation": "StandardScaler transforms data to have mean=0 and standard deviation=1."
                    },
                    {
                        "question": "Which algorithms are most sensitive to feature scaling?",
                        "options": ["Decision Trees", "Random Forest", "K-NN and SVM", "Naive Bayes"],
                        "correct": 2,
                        "explanation": "Distance-based algorithms like K-NN and SVM are very sensitive to feature scaling."
                    },
                    {
                        "question": "What is RobustScaler best used for?",
                        "options": ["Normal data", "Data with outliers", "Categorical data", "Time series data"],
                        "correct": 1,
                        "explanation": "RobustScaler uses median and IQR, making it robust to outliers."
                    },
                    {
                        "question": "When should you apply feature scaling?",
                        "options": ["Before train-test split", "After train-test split", "Only on training data", "Only on test data"],
                        "correct": 1,
                        "explanation": "Fit scaler on training data, then transform both training and test data to prevent data leakage."
                    }
                ]
            },
            
            # Regression
            "Linear Regression": {
                "problems": [
                    {
                        "question": "What assumption does linear regression make about the relationship between variables?",
                        "options": ["Exponential relationship", "Linear relationship", "Logarithmic relationship", "No specific relationship"],
                        "correct": 1,
                        "explanation": "Linear regression assumes a linear relationship between independent and dependent variables."
                    },
                    {
                        "question": "Which metric is most appropriate for evaluating linear regression?",
                        "options": ["Accuracy", "F1-score", "RMSE", "Precision"],
                        "correct": 2,
                        "explanation": "RMSE (Root Mean Square Error) measures the average prediction error in the same units as the target variable."
                    },
                    {
                        "question": "What does R² (R-squared) measure?",
                        "options": ["Error rate", "Variance explained", "Correlation", "Bias"],
                        "correct": 1,
                        "explanation": "R² measures the proportion of variance in the dependent variable explained by the model."
                    },
                    {
                        "question": "What is multicollinearity in linear regression?",
                        "options": ["Multiple targets", "Correlated features", "Multiple models", "Non-linear relationships"],
                        "correct": 1,
                        "explanation": "Multicollinearity occurs when independent variables are highly correlated with each other."
                    },
                    {
                        "question": "Which assumption is violated when residuals show a pattern?",
                        "options": ["Linearity", "Independence", "Homoscedasticity", "Normality"],
                        "correct": 2,
                        "explanation": "Patterned residuals indicate heteroscedasticity (non-constant variance)."
                    }
                ]
            },
            
            "Polynomial Regression": {
                "problems": [
                    {
                        "question": "What is the main advantage of polynomial regression over linear regression?",
                        "options": ["Faster training", "Can model non-linear relationships", "Less overfitting", "Simpler interpretation"],
                        "correct": 1,
                        "explanation": "Polynomial regression can capture non-linear relationships by adding polynomial terms."
                    },
                    {
                        "question": "What is the main risk of using high-degree polynomials?",
                        "options": ["Underfitting", "Overfitting", "Slow training", "Poor interpretability"],
                        "correct": 1,
                        "explanation": "High-degree polynomials can overfit to training data, especially with limited samples."
                    },
                    {
                        "question": "How do you choose the optimal polynomial degree?",
                        "options": ["Always use degree 2", "Cross-validation", "Use highest degree", "Random selection"],
                        "correct": 1,
                        "explanation": "Cross-validation helps find the degree that balances bias and variance."
                    },
                    {
                        "question": "What happens to model complexity as polynomial degree increases?",
                        "options": ["Decreases", "Stays same", "Increases", "Becomes unpredictable"],
                        "correct": 2,
                        "explanation": "Higher polynomial degrees increase model complexity and flexibility."
                    },
                    {
                        "question": "Which technique can help prevent overfitting in polynomial regression?",
                        "options": ["More data", "Regularization", "Feature scaling", "All of the above"],
                        "correct": 3,
                        "explanation": "More data, regularization, and proper preprocessing all help prevent overfitting."
                    }
                ]
            },
            
            "SVR": {
                "problems": [
                    {
                        "question": "What is the main advantage of Support Vector Regression (SVR)?",
                        "options": ["Fast training", "Works well with high-dimensional data", "Simple interpretation", "No hyperparameters"],
                        "correct": 1,
                        "explanation": "SVR works well with high-dimensional data and is effective when the number of features exceeds the number of samples."
                    },
                    {
                        "question": "What does the epsilon parameter control in SVR?",
                        "options": ["Regularization strength", "Width of the margin", "Learning rate", "Number of support vectors"],
                        "correct": 1,
                        "explanation": "Epsilon defines the width of the epsilon-insensitive tube around the regression line."
                    },
                    {
                        "question": "Which kernel is most commonly used in SVR for non-linear relationships?",
                        "options": ["Linear", "Polynomial", "RBF (Gaussian)", "Sigmoid"],
                        "correct": 2,
                        "explanation": "RBF (Radial Basis Function) kernel is most commonly used for capturing non-linear relationships."
                    },
                    {
                        "question": "What happens when you increase the C parameter in SVR?",
                        "options": ["More regularization", "Less regularization", "No effect", "Changes kernel"],
                        "correct": 1,
                        "explanation": "Higher C values reduce regularization, allowing the model to fit training data more closely."
                    },
                    {
                        "question": "What are support vectors in SVR?",
                        "options": ["All training points", "Points outside the epsilon tube", "Centroids", "Outliers"],
                        "correct": 1,
                        "explanation": "Support vectors are the training points that lie outside the epsilon-insensitive tube."
                    }
                ]
            },
            
            "Decision Tree Regression": {
                "problems": [
                    {
                        "question": "How does a decision tree make predictions for regression?",
                        "options": ["Weighted average", "Mean of leaf node values", "Median of leaf node", "Mode of leaf node"],
                        "correct": 1,
                        "explanation": "Decision trees predict the mean value of the target variable in the leaf node."
                    },
                    {
                        "question": "What criterion is used for splitting in regression trees?",
                        "options": ["Gini impurity", "Mean Squared Error", "Entropy", "Information gain"],
                        "correct": 1,
                        "explanation": "Regression trees use Mean Squared Error (MSE) as the splitting criterion."
                    },
                    {
                        "question": "What is the main disadvantage of decision trees?",
                        "options": ["Slow training", "Prone to overfitting", "Can't handle categorical data", "Requires feature scaling"],
                        "correct": 1,
                        "explanation": "Decision trees are prone to overfitting, especially with deep trees and small datasets."
                    },
                    {
                        "question": "How can you prevent overfitting in decision trees?",
                        "options": ["Increase max_depth", "Pruning", "Add more features", "Use more data only"],
                        "correct": 1,
                        "explanation": "Pruning (pre-pruning or post-pruning) helps prevent overfitting by limiting tree complexity."
                    },
                    {
                        "question": "What is the advantage of decision trees over linear regression?",
                        "options": ["Always more accurate", "Can capture non-linear relationships", "Faster training", "Better interpretability"],
                        "correct": 1,
                        "explanation": "Decision trees can capture non-linear relationships and interactions between features."
                    }
                ]
            },
            
            "Random Forest Regression": {
                "problems": [
                    {
                        "question": "What is the main advantage of Random Forest over a single Decision Tree?",
                        "options": ["Faster training", "Reduces overfitting", "Uses less memory", "Simpler interpretation"],
                        "correct": 1,
                        "explanation": "Random Forest reduces overfitting by averaging predictions from multiple decision trees."
                    },
                    {
                        "question": "How does Random Forest introduce randomness?",
                        "options": ["Random data", "Random features and samples", "Random parameters", "Random initialization"],
                        "correct": 1,
                        "explanation": "Random Forest uses random subsets of features and bootstrap samples for each tree."
                    },
                    {
                        "question": "What is bootstrap sampling in Random Forest?",
                        "options": ["Sampling without replacement", "Sampling with replacement", "Stratified sampling", "Systematic sampling"],
                        "correct": 1,
                        "explanation": "Bootstrap sampling draws samples with replacement to create diverse training sets."
                    },
                    {
                        "question": "How does Random Forest make final predictions for regression?",
                        "options": ["Takes the mode", "Takes the average", "Takes the maximum", "Uses weighted voting"],
                        "correct": 1,
                        "explanation": "Random Forest averages predictions from all trees for regression tasks."
                    },
                    {
                        "question": "What is feature importance in Random Forest?",
                        "options": ["Feature correlation", "Feature variance", "Contribution to splits", "Feature frequency"],
                        "correct": 2,
                        "explanation": "Feature importance measures how much each feature contributes to decreasing impurity across all trees."
                    }
                ]
            },
            
            # Classification
            "Random Forest Classification": {
                "problems": [
                    {
                        "question": "How does Random Forest make final predictions for classification?",
                        "options": ["Takes the average", "Uses majority voting", "Takes the maximum", "Uses weighted average"],
                        "correct": 1,
                        "explanation": "Random Forest uses majority voting for classification tasks."
                    },
                    {
                        "question": "What is out-of-bag (OOB) error in Random Forest?",
                        "options": ["Training error", "Validation error without separate validation set", "Test error", "Cross-validation error"],
                        "correct": 1,
                        "explanation": "OOB error uses samples not included in bootstrap sampling to estimate model performance."
                    },
                    {
                        "question": "Which hyperparameter controls the randomness in feature selection?",
                        "options": ["n_estimators", "max_features", "max_depth", "min_samples_split"],
                        "correct": 1,
                        "explanation": "max_features determines how many features are randomly selected at each split."
                    },
                    {
                        "question": "What happens when you increase n_estimators in Random Forest?",
                        "options": ["Always improves performance", "May improve performance but increases computation", "Always decreases performance", "No effect"],
                        "correct": 1,
                        "explanation": "More trees generally improve performance but with diminishing returns and increased computation."
                    },
                    {
                        "question": "Why is Random Forest less prone to overfitting than individual trees?",
                        "options": ["Uses less data", "Ensemble averaging", "Simpler model", "Better regularization"],
                        "correct": 1,
                        "explanation": "Ensemble averaging reduces variance and overfitting compared to individual trees."
                    }
                ]
            },
            
            "Logistic Regression": {
                "problems": [
                    {
                        "question": "What function does logistic regression use to map any real number to a probability?",
                        "options": ["Linear function", "Sigmoid function", "ReLU function", "Exponential function"],
                        "correct": 1,
                        "explanation": "The sigmoid function maps any real number to a value between 0 and 1, representing probability."
                    },
                    {
                        "question": "What's the decision boundary for logistic regression?",
                        "options": ["Curved line", "Straight line", "Circle", "Parabola"],
                        "correct": 1,
                        "explanation": "Logistic regression creates a linear decision boundary in the feature space."
                    },
                    {
                        "question": "What is the cost function used in logistic regression?",
                        "options": ["Mean Squared Error", "Log-likelihood", "Hinge loss", "Absolute error"],
                        "correct": 1,
                        "explanation": "Logistic regression uses log-likelihood (or cross-entropy) as its cost function."
                    },
                    {
                        "question": "What does the coefficient in logistic regression represent?",
                        "options": ["Probability", "Log-odds ratio", "Correlation", "Variance"],
                        "correct": 1,
                        "explanation": "Coefficients represent the change in log-odds for a unit change in the feature."
                    },
                    {
                        "question": "How do you extend logistic regression to multi-class problems?",
                        "options": ["Use multiple models", "One-vs-Rest or Multinomial", "Change activation function", "Add more features"],
                        "correct": 1,
                        "explanation": "Multi-class logistic regression uses One-vs-Rest or multinomial (softmax) approaches."
                    }
                ]
            },
            
            "K-NN": {
                "problems": [
                    {
                        "question": "What happens if you choose K=1 in K-NN?",
                        "options": ["High bias, low variance", "Low bias, high variance", "Balanced bias-variance", "No prediction possible"],
                        "correct": 1,
                        "explanation": "K=1 leads to low bias but high variance, making the model sensitive to noise."
                    },
                    {
                        "question": "How do you choose the optimal K in K-NN?",
                        "options": ["Always use K=3", "Cross-validation", "Use square root of n", "Random selection"],
                        "correct": 1,
                        "explanation": "Cross-validation helps find the K that gives the best performance on validation data."
                    },
                    {
                        "question": "What is the main computational disadvantage of K-NN?",
                        "options": ["High training time", "High prediction time", "High memory usage", "Complex implementation"],
                        "correct": 1,
                        "explanation": "K-NN has high prediction time as it needs to compute distances to all training points."
                    },
                    {
                        "question": "Why is feature scaling important for K-NN?",
                        "options": ["Improves accuracy", "Distance calculations are affected by scale", "Reduces overfitting", "Speeds up training"],
                        "correct": 1,
                        "explanation": "Features with larger scales dominate distance calculations, affecting neighbor selection."
                    },
                    {
                        "question": "What distance metric is commonly used in K-NN?",
                        "options": ["Manhattan distance", "Euclidean distance", "Cosine similarity", "All of the above"],
                        "correct": 3,
                        "explanation": "K-NN can use various distance metrics depending on the data type and problem."
                    }
                ]
            },
            
            "SVM": {
                "problems": [
                    {
                        "question": "What is the main objective of SVM?",
                        "options": ["Minimize error", "Maximize margin", "Minimize complexity", "Maximize accuracy"],
                        "correct": 1,
                        "explanation": "SVM aims to find the hyperplane that maximizes the margin between different classes."
                    },
                    {
                        "question": "What are support vectors?",
                        "options": ["All training points", "Points closest to the decision boundary", "Centroids", "Outliers"],
                        "correct": 1,
                        "explanation": "Support vectors are the data points closest to the decision boundary that define the margin."
                    },
                    {
                        "question": "What does the C parameter control in SVM?",
                        "options": ["Kernel type", "Trade-off between margin and misclassification", "Number of support vectors", "Learning rate"],
                        "correct": 1,
                        "explanation": "C parameter controls the trade-off between maximizing margin and minimizing classification errors."
                    },
                    {
                        "question": "Which kernel allows SVM to handle non-linearly separable data?",
                        "options": ["Linear kernel", "Polynomial kernel", "RBF kernel", "Both B and C"],
                        "correct": 3,
                        "explanation": "Both polynomial and RBF kernels can map data to higher dimensions for non-linear separation."
                    },
                    {
                        "question": "What is the kernel trick in SVM?",
                        "options": ["Fast training method", "Computing dot products in higher dimensions", "Feature selection", "Regularization technique"],
                        "correct": 1,
                        "explanation": "The kernel trick allows computing dot products in higher-dimensional space without explicitly mapping data."
                    }
                ]
            },
            
            "Naive Bayes": {
                "problems": [
                    {
                        "question": "What assumption does Naive Bayes make about features?",
                        "options": ["Features are correlated", "Features are independent", "Features are normally distributed", "Features are categorical"],
                        "correct": 1,
                        "explanation": "Naive Bayes assumes that features are conditionally independent given the class label."
                    },
                    {
                        "question": "Which theorem is Naive Bayes based on?",
                        "options": ["Central Limit Theorem", "Bayes' Theorem", "Law of Large Numbers", "Chebyshev's Theorem"],
                        "correct": 1,
                        "explanation": "Naive Bayes is based on Bayes' Theorem for calculating posterior probabilities."
                    },
                    {
                        "question": "What type of data is Gaussian Naive Bayes best suited for?",
                        "options": ["Categorical data", "Continuous numerical data", "Text data", "Binary data"],
                        "correct": 1,
                        "explanation": "Gaussian Naive Bayes assumes features follow a normal distribution, making it suitable for continuous data."
                    },
                    {
                        "question": "What is Laplace smoothing in Naive Bayes?",
                        "options": ["Feature scaling", "Handling zero probabilities", "Regularization", "Dimensionality reduction"],
                        "correct": 1,
                        "explanation": "Laplace smoothing adds a small constant to avoid zero probabilities for unseen feature-class combinations."
                    },
                    {
                        "question": "Why does Naive Bayes work well for text classification despite the independence assumption?",
                        "options": ["Text features are actually independent", "The assumption doesn't matter", "It's computationally efficient", "It handles high-dimensional data well"],
                        "correct": 3,
                        "explanation": "Despite violated independence assumptions, Naive Bayes performs well in high-dimensional spaces like text classification."
                    }
                ]
            },
            
            "Decision Tree Classification": {
                "problems": [
                    {
                        "question": "What criterion is commonly used for splitting in classification trees?",
                        "options": ["Mean Squared Error", "Gini impurity", "R-squared", "Mean Absolute Error"],
                        "correct": 1,
                        "explanation": "Gini impurity measures the probability of misclassifying a randomly chosen element."
                    },
                    {
                        "question": "What does entropy measure in decision trees?",
                        "options": ["Tree depth", "Information content/disorder", "Accuracy", "Number of nodes"],
                        "correct": 1,
                        "explanation": "Entropy measures the amount of information or disorder in a set of class labels."
                    },
                    {
                        "question": "What is information gain?",
                        "options": ["Accuracy improvement", "Reduction in entropy after splitting", "Number of correct predictions", "Tree complexity"],
                        "correct": 1,
                        "explanation": "Information gain measures the reduction in entropy achieved by splitting on a particular feature."
                    },
                    {
                        "question": "How do you handle overfitting in decision trees?",
                        "options": ["Increase tree depth", "Set minimum samples per leaf", "Use all features", "Remove pruning"],
                        "correct": 1,
                        "explanation": "Setting minimum samples per leaf prevents the tree from creating very specific rules for small groups."
                    },
                    {
                        "question": "What is the main advantage of decision trees for interpretation?",
                        "options": ["High accuracy", "Fast prediction", "Easy to visualize and understand", "Handles missing values"],
                        "correct": 2,
                        "explanation": "Decision trees create interpretable rules that can be easily visualized and understood by humans."
                    }
                ]
            },
            
            # Clustering
            "K-Means": {
                "problems": [
                    {
                        "question": "How do you choose the optimal number of clusters in K-Means?",
                        "options": ["Always use 3", "Use the elbow method", "Use cross-validation", "Random selection"],
                        "correct": 1,
                        "explanation": "The elbow method plots within-cluster sum of squares vs number of clusters to find the optimal K."
                    },
                    {
                        "question": "What is a limitation of K-Means clustering?",
                        "options": ["Can't handle large datasets", "Assumes spherical clusters", "Only works with 2D data", "Requires labeled data"],
                        "correct": 1,
                        "explanation": "K-Means assumes clusters are spherical and may struggle with non-spherical cluster shapes."
                    },
                    {
                        "question": "What does K-Means minimize?",
                        "options": ["Between-cluster variance", "Within-cluster sum of squares", "Total variance", "Number of iterations"],
                        "correct": 1,
                        "explanation": "K-Means minimizes the within-cluster sum of squares (WCSS)."
                    },
                    {
                        "question": "How are initial centroids typically chosen in K-Means?",
                        "options": ["Randomly", "K-Means++", "Fixed positions", "User-defined"],
                        "correct": 1,
                        "explanation": "K-Means++ initialization chooses initial centroids to be far apart, improving convergence."
                    },
                    {
                        "question": "What happens if you choose K equal to the number of data points?",
                        "options": ["Perfect clustering", "Each point is its own cluster", "Algorithm fails", "Infinite loop"],
                        "correct": 1,
                        "explanation": "Each data point becomes its own cluster, resulting in zero within-cluster variance."
                    }
                ]
            },
            
            "Hierarchical Clustering": {
                "problems": [
                    {
                        "question": "What is the main advantage of hierarchical clustering over K-Means?",
                        "options": ["Faster computation", "Don't need to specify K", "Better for large datasets", "Always finds global optimum"],
                        "correct": 1,
                        "explanation": "Hierarchical clustering doesn't require pre-specifying the number of clusters."
                    },
                    {
                        "question": "What is a dendrogram?",
                        "options": ["Cluster center", "Tree-like diagram showing cluster hierarchy", "Distance matrix", "Similarity measure"],
                        "correct": 1,
                        "explanation": "A dendrogram visualizes the hierarchical relationship between clusters."
                    },
                    {
                        "question": "What is the difference between agglomerative and divisive clustering?",
                        "options": ["Speed difference", "Bottom-up vs top-down", "Accuracy difference", "Distance metric"],
                        "correct": 1,
                        "explanation": "Agglomerative starts with individual points and merges (bottom-up), divisive starts with all points and splits (top-down)."
                    },
                    {
                        "question": "What is linkage criteria in hierarchical clustering?",
                        "options": ["Number of clusters", "How to measure distance between clusters", "Stopping condition", "Initialization method"],
                        "correct": 1,
                        "explanation": "Linkage criteria (single, complete, average) determines how distance between clusters is calculated."
                    },
                    {
                        "question": "What is the time complexity of hierarchical clustering?",
                        "options": ["O(n)", "O(n log n)", "O(n²)", "O(n³)"],
                        "correct": 3,
                        "explanation": "Hierarchical clustering has O(n³) time complexity, making it slow for large datasets."
                    }
                ]
            },
            
            "Apriori": {
                "problems": [
                    {
                        "question": "What does the Apriori algorithm find?",
                        "options": ["Clusters", "Classifications", "Frequent itemsets", "Regression coefficients"],
                        "correct": 2,
                        "explanation": "Apriori algorithm finds frequent itemsets in transactional data for association rule mining."
                    },
                    {
                        "question": "What is the Apriori principle?",
                        "options": ["All subsets of frequent itemsets are frequent", "All supersets are infrequent", "Items are independent", "Transactions are ordered"],
                        "correct": 0,
                        "explanation": "The Apriori principle states that all subsets of a frequent itemset must also be frequent."
                    },
                    {
                        "question": "What does 'support' measure in association rules?",
                        "options": ["Rule accuracy", "Frequency of itemset occurrence", "Rule strength", "Confidence level"],
                        "correct": 1,
                        "explanation": "Support measures how frequently an itemset appears in the dataset."
                    },
                    {
                        "question": "What does 'confidence' measure in association rules?",
                        "options": ["Frequency of itemset", "Conditional probability of consequent given antecedent", "Rule importance", "Statistical significance"],
                        "correct": 1,
                        "explanation": "Confidence measures the probability of finding the consequent given the antecedent."
                    },
                    {
                        "question": "What is 'lift' in association rule mining?",
                        "options": ["Support ratio", "Confidence ratio", "How much more likely items occur together vs independently", "Rule strength"],
                        "correct": 2,
                        "explanation": "Lift measures how much more likely items are to be bought together compared to being bought independently."
                    }
                ]
            },
            
            "Eclat": {
                "problems": [
                    {
                        "question": "How does Eclat differ from Apriori?",
                        "options": ["Uses different data structure", "Uses vertical data format", "Faster for sparse data", "All of the above"],
                        "correct": 3,
                        "explanation": "Eclat uses vertical data format (transaction IDs for each item) making it more efficient for sparse datasets."
                    },
                    {
                        "question": "What data structure does Eclat use?",
                        "options": ["Horizontal itemsets", "Vertical transaction lists", "Hash tables", "Decision trees"],
                        "correct": 1,
                        "explanation": "Eclat uses vertical transaction lists where each item maps to a list of transaction IDs."
                    },
                    {
                        "question": "What is the main advantage of Eclat over Apriori?",
                        "options": ["Better accuracy", "No database scans needed", "Handles categorical data", "More interpretable"],
                        "correct": 1,
                        "explanation": "Eclat avoids repeated database scans by using intersection operations on transaction ID lists."
                    },
                    {
                        "question": "How does Eclat calculate support?",
                        "options": ["Counting occurrences", "Intersection of transaction ID lists", "Statistical calculation", "Probability estimation"],
                        "correct": 1,
                        "explanation": "Eclat calculates support by finding the intersection of transaction ID lists for items."
                    },
                    {
                        "question": "When is Eclat preferred over Apriori?",
                        "options": ["Dense datasets", "Sparse datasets with many items", "Small datasets", "Categorical data only"],
                        "correct": 1,
                        "explanation": "Eclat is more efficient for sparse datasets with many items due to its vertical data representation."
                    }
                ]
            },
            
            "UCB": {
                "problems": [
                    {
                        "question": "What does UCB stand for in reinforcement learning?",
                        "options": ["Upper Confidence Bound", "Uniform Confidence Bound", "Universal Control Bound", "Updated Confidence Bound"],
                        "correct": 0,
                        "explanation": "UCB stands for Upper Confidence Bound, a strategy for the multi-armed bandit problem."
                    },
                    {
                        "question": "What problem does UCB solve?",
                        "options": ["Classification", "Regression", "Multi-armed bandit", "Clustering"],
                        "correct": 2,
                        "explanation": "UCB solves the multi-armed bandit problem by balancing exploration and exploitation."
                    },
                    {
                        "question": "How does UCB balance exploration and exploitation?",
                        "options": ["Random selection", "Confidence intervals", "Probability distributions", "Gradient descent"],
                        "correct": 1,
                        "explanation": "UCB uses confidence intervals to select actions that balance known rewards with uncertainty."
                    },
                    {
                        "question": "What happens to the confidence bound as more samples are collected?",
                        "options": ["Increases", "Decreases", "Stays constant", "Becomes random"],
                        "correct": 1,
                        "explanation": "The confidence bound decreases as more samples reduce uncertainty about the action's value."
                    },
                    {
                        "question": "What is the regret in multi-armed bandit problems?",
                        "options": ["Prediction error", "Difference from optimal strategy", "Training loss", "Validation error"],
                        "correct": 1,
                        "explanation": "Regret measures the difference between the reward obtained and the optimal possible reward."
                    }
                ]
            },
            
            "Thompson Sampling": {
                "problems": [
                    {
                        "question": "What approach does Thompson Sampling use?",
                        "options": ["Deterministic selection", "Bayesian approach with probability sampling", "Greedy selection", "Random selection"],
                        "correct": 1,
                        "explanation": "Thompson Sampling uses Bayesian inference to maintain probability distributions over action values."
                    },
                    {
                        "question": "How does Thompson Sampling select actions?",
                        "options": ["Highest mean reward", "Sampling from posterior distributions", "Random selection", "Confidence intervals"],
                        "correct": 1,
                        "explanation": "Thompson Sampling selects actions by sampling from the posterior distribution of each action's value."
                    },
                    {
                        "question": "What is the main advantage of Thompson Sampling over UCB?",
                        "options": ["Faster computation", "Better theoretical guarantees", "Natural handling of prior knowledge", "Simpler implementation"],
                        "correct": 2,
                        "explanation": "Thompson Sampling naturally incorporates prior knowledge through Bayesian updating."
                    },
                    {
                        "question": "What distribution is commonly used for binary rewards in Thompson Sampling?",
                        "options": ["Normal distribution", "Beta distribution", "Uniform distribution", "Exponential distribution"],
                        "correct": 1,
                        "explanation": "Beta distribution is the conjugate prior for Bernoulli likelihood, making it ideal for binary rewards."
                    },
                    {
                        "question": "How does Thompson Sampling handle the exploration-exploitation trade-off?",
                        "options": ["Explicit exploration parameter", "Naturally through uncertainty in posterior", "Random exploration", "Fixed exploration rate"],
                        "correct": 1,
                        "explanation": "Thompson Sampling naturally balances exploration and exploitation through uncertainty in posterior distributions."
                    }
                ]
            },
            
            "Q-Learning": {
                "problems": [
                    {
                        "question": "What does Q-Learning learn?",
                        "options": ["State values", "Action-value function", "Policy directly", "Reward function"],
                        "correct": 1,
                        "explanation": "Q-Learning learns the action-value function Q(s,a) representing expected future rewards."
                    },
                    {
                        "question": "What is the Q-Learning update rule based on?",
                        "options": ["Gradient descent", "Bellman equation", "Maximum likelihood", "Least squares"],
                        "correct": 1,
                        "explanation": "Q-Learning uses the Bellman equation to update Q-values based on observed rewards and future estimates."
                    },
                    {
                        "question": "What does the learning rate (alpha) control in Q-Learning?",
                        "options": ["Exploration rate", "How much new information updates Q-values", "Discount factor", "Convergence speed"],
                        "correct": 1,
                        "explanation": "The learning rate controls how much the new information updates the existing Q-value estimates."
                    },
                    {
                        "question": "What is the epsilon-greedy strategy in Q-Learning?",
                        "options": ["Always choose best action", "Balance exploration and exploitation", "Random action selection", "Greedy action selection"],
                        "correct": 1,
                        "explanation": "Epsilon-greedy chooses the best action with probability (1-ε) and explores randomly with probability ε."
                    },
                    {
                        "question": "What is the discount factor (gamma) in Q-Learning?",
                        "options": ["Learning rate", "Exploration rate", "Importance of future rewards", "Convergence parameter"],
                        "correct": 2,
                        "explanation": "The discount factor determines how much future rewards are valued compared to immediate rewards."
                    }
                ]
            },
            
            "Bag-of-Words": {
                "problems": [
                    {
                        "question": "What does the Bag-of-Words model represent?",
                        "options": ["Word order", "Word frequency", "Word meaning", "Word relationships"],
                        "correct": 1,
                        "explanation": "Bag-of-Words represents text as a collection of word frequencies, ignoring grammar and word order."
                    },
                    {
                        "question": "What information does Bag-of-Words lose?",
                        "options": ["Word frequency", "Vocabulary size", "Word order and context", "Document length"],
                        "correct": 2,
                        "explanation": "Bag-of-Words loses word order, grammar, and contextual relationships between words."
                    },
                    {
                        "question": "How is the vocabulary size determined in Bag-of-Words?",
                        "options": ["Fixed size", "Number of unique words in corpus", "Document length", "Sentence count"],
                        "correct": 1,
                        "explanation": "Vocabulary size equals the number of unique words across all documents in the corpus."
                    },
                    {
                        "question": "What is a major limitation of Bag-of-Words for large vocabularies?",
                        "options": ["Slow training", "High dimensionality and sparsity", "Poor accuracy", "Memory issues only"],
                        "correct": 1,
                        "explanation": "Large vocabularies create high-dimensional, sparse feature vectors that are computationally challenging."
                    },
                    {
                        "question": "How can you reduce dimensionality in Bag-of-Words?",
                        "options": ["Remove stop words", "Use stemming", "Set minimum frequency threshold", "All of the above"],
                        "correct": 3,
                        "explanation": "All these techniques help reduce vocabulary size and dimensionality in Bag-of-Words representation."
                    }
                ]
            },
            
            "TF-IDF": {
                "problems": [
                    {
                        "question": "What does TF-IDF stand for?",
                        "options": ["Term Frequency-Inverse Document Frequency", "Text Frequency-Index Document Frequency", "Token Frequency-Inverse Data Frequency", "Term Filter-Inverse Document Filter"],
                        "correct": 0,
                        "explanation": "TF-IDF stands for Term Frequency-Inverse Document Frequency."
                    },
                    {
                        "question": "What does the IDF component measure?",
                        "options": ["Word frequency in document", "Word importance across corpus", "Document length", "Vocabulary size"],
                        "correct": 1,
                        "explanation": "IDF measures how important a word is across the entire corpus by penalizing common words."
                    },
                    {
                        "question": "Why is IDF useful in text analysis?",
                        "options": ["Increases word frequency", "Reduces impact of common words", "Improves grammar", "Maintains word order"],
                        "correct": 1,
                        "explanation": "IDF reduces the impact of common words that appear in many documents and aren't discriminative."
                    },
                    {
                        "question": "How is TF-IDF calculated?",
                        "options": ["TF + IDF", "TF - IDF", "TF × IDF", "TF / IDF"],
                        "correct": 2,
                        "explanation": "TF-IDF is calculated as the product of Term Frequency and Inverse Document Frequency."
                    },
                    {
                        "question": "What advantage does TF-IDF have over simple word counts?",
                        "options": ["Faster computation", "Better handles document length differences", "Preserves word order", "Reduces vocabulary size"],
                        "correct": 1,
                        "explanation": "TF-IDF normalizes for document length and emphasizes distinctive words over common ones."
                    }
                ]
            },
            
            "Word2Vec": {
                "problems": [
                    {
                        "question": "What does Word2Vec create?",
                        "options": ["Word frequencies", "Dense vector representations of words", "Document classifications", "Grammar rules"],
                        "correct": 1,
                        "explanation": "Word2Vec creates dense, low-dimensional vector representations that capture semantic relationships."
                    },
                    {
                        "question": "What are the two main architectures of Word2Vec?",
                        "options": ["CBOW and Skip-gram", "TF-IDF and Bag-of-Words", "RNN and LSTM", "Encoder and Decoder"],
                        "correct": 0,
                        "explanation": "Word2Vec has two architectures: CBOW (Continuous Bag of Words) and Skip-gram."
                    },
                    {
                        "question": "How does Skip-gram work?",
                        "options": ["Predicts word from context", "Predicts context from word", "Counts word frequency", "Analyzes grammar"],
                        "correct": 1,
                        "explanation": "Skip-gram predicts surrounding context words given a target word."
                    },
                    {
                        "question": "What is a key advantage of Word2Vec over Bag-of-Words?",
                        "options": ["Faster training", "Captures semantic relationships", "Smaller vocabulary", "Better for short texts"],
                        "correct": 1,
                        "explanation": "Word2Vec captures semantic relationships, allowing operations like 'king - man + woman = queen'."
                    },
                    {
                        "question": "What is negative sampling in Word2Vec?",
                        "options": ["Removing bad words", "Training efficiency technique", "Error correction", "Data cleaning"],
                        "correct": 1,
                        "explanation": "Negative sampling is a technique to make Word2Vec training more efficient by sampling negative examples."
                    }
                ]
            },
            
            "BERT": {
                "problems": [
                    {
                        "question": "What does BERT stand for?",
                        "options": ["Basic Encoder Representation Transformer", "Bidirectional Encoder Representations from Transformers", "Binary Encoder Recurrent Transformer", "Balanced Encoder Representation Technique"],
                        "correct": 1,
                        "explanation": "BERT stands for Bidirectional Encoder Representations from Transformers."
                    },
                    {
                        "question": "What makes BERT bidirectional?",
                        "options": ["Processes text forwards and backwards", "Uses two separate models", "Has two attention mechanisms", "Trains on reversed text"],
                        "correct": 0,
                        "explanation": "BERT processes text in both directions simultaneously, considering both left and right context."
                    },
                    {
                        "question": "What are the two pre-training tasks in BERT?",
                        "options": ["Classification and regression", "Masked Language Model and Next Sentence Prediction", "Encoding and decoding", "Attention and feedforward"],
                        "correct": 1,
                        "explanation": "BERT uses Masked Language Model (MLM) and Next Sentence Prediction (NSP) for pre-training."
                    },
                    {
                        "question": "What is the Masked Language Model task?",
                        "options": ["Removing stop words", "Predicting masked words from context", "Translating languages", "Summarizing text"],
                        "correct": 1,
                        "explanation": "MLM randomly masks words in sentences and trains the model to predict the masked words."
                    },
                    {
                        "question": "How is BERT typically used for downstream tasks?",
                        "options": ["From scratch training", "Fine-tuning pre-trained model", "Feature extraction only", "Rule-based approach"],
                        "correct": 1,
                        "explanation": "BERT is typically fine-tuned on specific downstream tasks using the pre-trained representations."
                    }
                ]
            },
            
            "ANN": {
                "problems": [
                    {
                        "question": "What is the purpose of activation functions in neural networks?",
                        "options": ["Speed up training", "Introduce non-linearity", "Reduce overfitting", "Normalize inputs"],
                        "correct": 1,
                        "explanation": "Activation functions introduce non-linearity, allowing neural networks to learn complex patterns."
                    },
                    {
                        "question": "What is backpropagation?",
                        "options": ["Forward pass", "Algorithm to update weights", "Activation function", "Loss function"],
                        "correct": 1,
                        "explanation": "Backpropagation is the algorithm used to calculate gradients and update weights in neural networks."
                    },
                    {
                        "question": "What is the vanishing gradient problem?",
                        "options": ["Gradients become too large", "Gradients become too small", "No gradients", "Unstable gradients"],
                        "correct": 1,
                        "explanation": "Vanishing gradients occur when gradients become very small, making it hard to train deep networks."
                    },
                    {
                        "question": "Which activation function helps mitigate the vanishing gradient problem?",
                        "options": ["Sigmoid", "Tanh", "ReLU", "Linear"],
                        "correct": 2,
                        "explanation": "ReLU activation function helps mitigate vanishing gradients by maintaining constant gradients for positive inputs."
                    },
                    {
                        "question": "What is dropout in neural networks?",
                        "options": ["Removing layers", "Regularization technique", "Activation function", "Optimization algorithm"],
                        "correct": 1,
                        "explanation": "Dropout randomly sets some neurons to zero during training to prevent overfitting."
                    }
                ]
            },
            
            "CNN": {
                "problems": [
                    {
                        "question": "What is the main purpose of convolutional layers in CNN?",
                        "options": ["Reduce overfitting", "Extract local features", "Increase model size", "Speed up training"],
                        "correct": 1,
                        "explanation": "Convolutional layers extract local features like edges, textures, and patterns from input data."
                    },
                    {
                        "question": "What does pooling do in CNNs?",
                        "options": ["Increases spatial dimensions", "Reduces spatial dimensions", "Adds non-linearity", "Normalizes data"],
                        "correct": 1,
                        "explanation": "Pooling reduces spatial dimensions while retaining important information, making the model more efficient."
                    },
                    {
                        "question": "What is the advantage of parameter sharing in CNNs?",
                        "options": ["Faster training", "Fewer parameters", "Translation invariance", "All of the above"],
                        "correct": 3,
                        "explanation": "Parameter sharing reduces parameters, enables translation invariance, and makes training more efficient."
                    },
                    {
                        "question": "What is the receptive field in CNNs?",
                        "options": ["Filter size", "Input region affecting one output", "Number of channels", "Pooling window"],
                        "correct": 1,
                        "explanation": "Receptive field is the region in the input that influences a particular output neuron."
                    },
                    {
                        "question": "Why are CNNs particularly effective for image processing?",
                        "options": ["Handle variable input sizes", "Exploit spatial locality", "Require less data", "Always more accurate"],
                        "correct": 1,
                        "explanation": "CNNs exploit spatial locality and hierarchical patterns that are natural in images."
                    }
                ]
            },
            
            "RNN": {
                "problems": [
                    {
                        "question": "What makes RNNs suitable for sequential data?",
                        "options": ["Parallel processing", "Memory of previous inputs", "Fixed input size", "Fast computation"],
                        "correct": 1,
                        "explanation": "RNNs have memory through hidden states that carry information from previous time steps."
                    },
                    {
                        "question": "What is the vanishing gradient problem in RNNs?",
                        "options": ["Gradients become too large", "Gradients become too small", "No gradients", "Unstable gradients"],
                        "correct": 1,
                        "explanation": "Vanishing gradients occur when gradients become very small, making it hard to learn long-term dependencies."
                    },
                    {
                        "question": "How do RNNs process sequences?",
                        "options": ["All at once", "One element at a time", "In random order", "In reverse order"],
                        "correct": 1,
                        "explanation": "RNNs process sequences one element at a time, updating hidden state at each step."
                    },
                    {
                        "question": "What is backpropagation through time (BPTT)?",
                        "options": ["Forward pass algorithm", "Training algorithm for RNNs", "Activation function", "Regularization technique"],
                        "correct": 1,
                        "explanation": "BPTT is the training algorithm that unfolds RNN through time to compute gradients."
                    },
                    {
                        "question": "What type of problems are RNNs commonly used for?",
                        "options": ["Image classification", "Time series and NLP", "Clustering", "Dimensionality reduction"],
                        "correct": 1,
                        "explanation": "RNNs are commonly used for time series forecasting, natural language processing, and sequential data."
                    }
                ]
            },
            
            "LSTM": {
                "problems": [
                    {
                        "question": "What problem do LSTMs solve compared to vanilla RNNs?",
                        "options": ["Faster training", "Long-term memory", "Smaller model size", "Better accuracy always"],
                        "correct": 1,
                        "explanation": "LSTMs solve the vanishing gradient problem, enabling learning of long-term dependencies."
                    },
                    {
                        "question": "What are the three gates in an LSTM cell?",
                        "options": ["Input, Output, Hidden", "Forget, Input, Output", "Memory, Update, Reset", "Forward, Backward, Update"],
                        "correct": 1,
                        "explanation": "LSTM has three gates: forget gate, input gate, and output gate that control information flow."
                    },
                    {
                        "question": "What does the forget gate do?",
                        "options": ["Adds new information", "Decides what to remove from cell state", "Controls output", "Updates hidden state"],
                        "correct": 1,
                        "explanation": "The forget gate decides what information to discard from the cell state."
                    },
                    {
                        "question": "What is the cell state in LSTM?",
                        "options": ["Hidden state", "Long-term memory", "Input vector", "Output vector"],
                        "correct": 1,
                        "explanation": "The cell state acts as the long-term memory that flows through the LSTM with minimal changes."
                    },
                    {
                        "question": "How does LSTM handle the vanishing gradient problem?",
                        "options": ["Larger learning rates", "Gated architecture with direct paths", "More layers", "Different activation functions"],
                        "correct": 1,
                        "explanation": "LSTM's gated architecture provides direct paths for gradients to flow, preventing vanishing."
                    }
                ]
            },
            
            # Dimensionality Reduction
            "PCA": {
                "problems": [
                    {
                        "question": "What does PCA maximize when finding principal components?",
                        "options": ["Correlation", "Variance", "Mean", "Standard deviation"],
                        "correct": 1,
                        "explanation": "PCA finds components that maximize the variance in the data."
                    },
                    {
                        "question": "What are principal components?",
                        "options": ["Original features", "Linear combinations of original features", "Cluster centers", "Outliers"],
                        "correct": 1,
                        "explanation": "Principal components are linear combinations of original features that capture maximum variance."
                    },
                    {
                        "question": "How do you choose the number of components in PCA?",
                        "options": ["Always use 2", "Explained variance ratio", "Cross-validation", "Random selection"],
                        "correct": 1,
                        "explanation": "Choose components that explain a desired percentage of total variance (e.g., 95%)."
                    },
                    {
                        "question": "What is the first principal component?",
                        "options": ["Most correlated feature", "Direction of maximum variance", "Largest eigenvalue", "Mean of features"],
                        "correct": 1,
                        "explanation": "The first principal component is the direction along which data varies the most."
                    },
                    {
                        "question": "Why should you standardize features before PCA?",
                        "options": ["Improves accuracy", "Features with larger scales dominate", "Reduces computation", "Required by algorithm"],
                        "correct": 1,
                        "explanation": "Without standardization, features with larger scales will dominate the principal components."
                    }
                ]
            },
            
            "LDA": {
                "problems": [
                    {
                        "question": "What is the main difference between PCA and LDA?",
                        "options": ["PCA is supervised, LDA is unsupervised", "LDA is supervised, PCA is unsupervised", "They are the same", "LDA is faster"],
                        "correct": 1,
                        "explanation": "LDA is supervised (uses class labels) while PCA is unsupervised (ignores class labels)."
                    },
                    {
                        "question": "What does LDA maximize?",
                        "options": ["Total variance", "Between-class variance / Within-class variance", "Correlation", "Accuracy"],
                        "correct": 1,
                        "explanation": "LDA maximizes the ratio of between-class variance to within-class variance for better class separation."
                    },
                    {
                        "question": "How many components can LDA extract at most?",
                        "options": ["Number of features", "Number of samples", "Number of classes - 1", "Unlimited"],
                        "correct": 2,
                        "explanation": "LDA can extract at most (number of classes - 1) components for optimal class separation."
                    },
                    {
                        "question": "When is LDA preferred over PCA?",
                        "options": ["Always", "When you have class labels and want discrimination", "For unsupervised learning", "For large datasets only"],
                        "correct": 1,
                        "explanation": "LDA is preferred when you have class labels and want to maximize class separability."
                    },
                    {
                        "question": "What assumption does LDA make about class distributions?",
                        "options": ["All classes have same mean", "All classes have same covariance", "Classes are independent", "Classes are binary"],
                        "correct": 1,
                        "explanation": "LDA assumes that all classes have the same covariance matrix (homoscedasticity)."
                    }
                ]
            },
            
            "Kernel PCA": {
                "problems": [
                    {
                        "question": "What is the main advantage of Kernel PCA over standard PCA?",
                        "options": ["Faster computation", "Can capture non-linear relationships", "Uses less memory", "Better for small datasets"],
                        "correct": 1,
                        "explanation": "Kernel PCA can capture non-linear relationships by mapping data to higher-dimensional space."
                    },
                    {
                        "question": "What is the kernel trick in Kernel PCA?",
                        "options": ["Fast computation method", "Computing dot products in higher dimensions without explicit mapping", "Data preprocessing", "Regularization technique"],
                        "correct": 1,
                        "explanation": "The kernel trick computes dot products in higher-dimensional space without explicitly mapping the data."
                    },
                    {
                        "question": "Which kernel is commonly used in Kernel PCA for non-linear data?",
                        "options": ["Linear kernel", "Polynomial kernel", "RBF kernel", "Both B and C"],
                        "correct": 3,
                        "explanation": "Both polynomial and RBF kernels are commonly used to capture non-linear patterns."
                    },
                    {
                        "question": "What is a disadvantage of Kernel PCA?",
                        "options": ["Cannot handle non-linear data", "Computationally expensive", "Only works with small datasets", "Requires labeled data"],
                        "correct": 1,
                        "explanation": "Kernel PCA is computationally expensive due to kernel matrix computation and eigendecomposition."
                    },
                    {
                        "question": "How do you choose the number of components in Kernel PCA?",
                        "options": ["Always use all components", "Based on eigenvalue magnitude", "Cross-validation", "Both B and C"],
                        "correct": 3,
                        "explanation": "Component selection can be based on eigenvalue magnitude or validated through cross-validation."
                    }
                ]
            },
            
            "Cross Validation": {
                "problems": [
                    {
                        "question": "What is the main purpose of cross-validation?",
                        "options": ["Speed up training", "Estimate model performance", "Reduce overfitting", "Feature selection"],
                        "correct": 1,
                        "explanation": "Cross-validation provides an unbiased estimate of model performance on unseen data."
                    },
                    {
                        "question": "What is k-fold cross-validation?",
                        "options": ["Training k different models", "Splitting data into k folds", "Using k features", "k iterations of training"],
                        "correct": 1,
                        "explanation": "K-fold CV splits data into k folds, using k-1 for training and 1 for validation, repeated k times."
                    },
                    {
                        "question": "What is leave-one-out cross-validation?",
                        "options": ["Remove one feature", "Use one sample for validation", "Remove one model", "Skip one iteration"],
                        "correct": 1,
                        "explanation": "Leave-one-out CV uses one sample for validation and the rest for training, repeated for each sample."
                    },
                    {
                        "question": "What is stratified cross-validation?",
                        "options": ["Random splitting", "Maintains class distribution in each fold", "Uses different algorithms", "Weighted sampling"],
                        "correct": 1,
                        "explanation": "Stratified CV ensures each fold maintains the same class distribution as the original dataset."
                    },
                    {
                        "question": "When should you use cross-validation?",
                        "options": ["Only for small datasets", "Model selection and performance estimation", "Only for classification", "Only for deep learning"],
                        "correct": 1,
                        "explanation": "Cross-validation is used for model selection, hyperparameter tuning, and performance estimation."
                    }
                ]
            },
            
            "Grid Search": {
                "problems": [
                    {
                        "question": "What is the purpose of Grid Search?",
                        "options": ["Feature selection", "Hyperparameter tuning", "Model training", "Data preprocessing"],
                        "correct": 1,
                        "explanation": "Grid Search systematically searches through hyperparameter combinations to find the best ones."
                    },
                    {
                        "question": "How does Grid Search work?",
                        "options": ["Random sampling", "Exhaustive search over parameter grid", "Gradient-based optimization", "Evolutionary algorithms"],
                        "correct": 1,
                        "explanation": "Grid Search performs exhaustive search over all combinations in the specified parameter grid."
                    },
                    {
                        "question": "What is a disadvantage of Grid Search?",
                        "options": ["Not thorough", "Computationally expensive", "Finds local optima", "Requires labeled data"],
                        "correct": 1,
                        "explanation": "Grid Search can be computationally expensive as it tests all parameter combinations."
                    },
                    {
                        "question": "How is Grid Search typically combined with cross-validation?",
                        "options": ["Sequential execution", "Each parameter combination is evaluated using CV", "Separate processes", "Not combined"],
                        "correct": 1,
                        "explanation": "Grid Search uses cross-validation to evaluate each parameter combination's performance."
                    },
                    {
                        "question": "What is Random Search as an alternative to Grid Search?",
                        "options": ["Searches randomly", "Samples parameter combinations randomly", "Uses random data", "Random model selection"],
                        "correct": 1,
                        "explanation": "Random Search randomly samples parameter combinations, often more efficient than exhaustive grid search."
                    }
                ]
            },
            
            "XGBoost": {
                "problems": [
                    {
                        "question": "What does XGBoost stand for?",
                        "options": ["eXtreme Gradient Boosting", "eXtended Gradient Boosting", "eXponential Gradient Boosting", "eXternal Gradient Boosting"],
                        "correct": 0,
                        "explanation": "XGBoost stands for eXtreme Gradient Boosting, an optimized gradient boosting framework."
                    },
                    {
                        "question": "What is the main advantage of XGBoost over traditional gradient boosting?",
                        "options": ["Simpler implementation", "Better performance and efficiency", "Requires less data", "Only works with trees"],
                        "correct": 1,
                        "explanation": "XGBoost provides better performance through regularization and computational optimizations."
                    },
                    {
                        "question": "What regularization techniques does XGBoost use?",
                        "options": ["L1 only", "L2 only", "Both L1 and L2", "No regularization"],
                        "correct": 2,
                        "explanation": "XGBoost uses both L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting."
                    },
                    {
                        "question": "How does XGBoost handle missing values?",
                        "options": ["Removes rows", "Mean imputation", "Learns optimal direction for missing values", "Requires preprocessing"],
                        "correct": 2,
                        "explanation": "XGBoost automatically learns the optimal direction to handle missing values during training."
                    },
                    {
                        "question": "What is early stopping in XGBoost?",
                        "options": ["Stopping after fixed iterations", "Stopping when validation performance stops improving", "Stopping when training is perfect", "Manual stopping"],
                        "correct": 1,
                        "explanation": "Early stopping prevents overfitting by stopping training when validation performance stops improving."
                    }
                ]
            },
            
            "AdaBoost": {
                "problems": [
                    {
                        "question": "What does AdaBoost stand for?",
                        "options": ["Adaptive Boosting", "Advanced Boosting", "Additive Boosting", "Automated Boosting"],
                        "correct": 0,
                        "explanation": "AdaBoost stands for Adaptive Boosting, which adaptively adjusts to errors of weak learners."
                    },
                    {
                        "question": "How does AdaBoost adjust to misclassified examples?",
                        "options": ["Removes them", "Increases their weights", "Decreases their weights", "Ignores them"],
                        "correct": 1,
                        "explanation": "AdaBoost increases weights of misclassified examples so subsequent learners focus on them."
                    },
                    {
                        "question": "What type of learners does AdaBoost typically use?",
                        "options": ["Strong learners", "Weak learners", "Random learners", "Complex learners"],
                        "correct": 1,
                        "explanation": "AdaBoost combines many weak learners (slightly better than random) to create a strong learner."
                    },
                    {
                        "question": "How does AdaBoost make final predictions?",
                        "options": ["Simple majority vote", "Weighted vote based on learner performance", "Average of predictions", "Last learner only"],
                        "correct": 1,
                        "explanation": "AdaBoost uses weighted voting where better-performing learners have higher weights."
                    },
                    {
                        "question": "What is a potential problem with AdaBoost?",
                        "options": ["Underfitting", "Sensitive to noise and outliers", "Too simple", "Requires large datasets"],
                        "correct": 1,
                        "explanation": "AdaBoost can be sensitive to noise and outliers as it focuses heavily on misclassified examples."
                    }
                ]
            }
        }
    
    def _initialize_coding_templates(self) -> Dict[str, Dict]:
        """Initialize comprehensive coding templates for all ML topics"""
        return {
            
            "Missing Data": {
                "Beginner": {
                    "problem": "Handle missing values in a customer dataset using different imputation strategies.",
                    "starter_code": """
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample dataset with missing values
data = {
    'age': [25, np.nan, 35, 45, np.nan],
    'income': [50000, 60000, np.nan, 80000, 55000],
    'category': ['A', 'B', np.nan, 'A', 'B']
}
df = pd.DataFrame(data)

# TODO: Handle missing numerical values using mean imputation
# TODO: Handle missing categorical values using mode imputation
# TODO: Display the cleaned dataset
""",
                    "solution": """
# Handle missing numerical values
num_imputer = SimpleImputer(strategy='mean')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])

# Handle missing categorical values
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['category']] = cat_imputer.fit_transform(df[['category']])

print("Cleaned dataset:")
print(df)
print(f"\\nMissing values: {df.isnull().sum().sum()}")
""",
                    "explanation": "This demonstrates different imputation strategies for numerical and categorical missing data."
                }
            },
            
            "Random Forest Classification": {
                "Beginner": {
                    "problem": "Build a Random Forest classifier to predict customer churn.",
                    "starter_code": """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample customer data
np.random.seed(42)
X = np.random.rand(1000, 4)  # features: age, income, usage, satisfaction
y = np.random.randint(0, 2, 1000)  # churn: 0=stay, 1=churn

# TODO: Split the data into training and testing sets
# TODO: Create and train a Random Forest classifier
# TODO: Make predictions and evaluate performance
""",
                    "solution": """
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = ['age', 'income', 'usage', 'satisfaction']
importance = rf_classifier.feature_importances_
for name, imp in zip(feature_names, importance):
    print(f"{name}: {imp:.3f}")
""",
                    "explanation": "This creates a Random Forest classifier for binary classification and shows feature importance."
                }
            },
            
            "Linear Regression": {
                "Beginner": {
                    "problem": "Create a simple linear regression model to predict house prices based on size.",
                    "starter_code": """
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: house sizes (sq ft) and prices ($1000s)
sizes = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
prices = np.array([100, 150, 200, 250, 300])

# TODO: Split the data into training and testing sets
# TODO: Create and train a linear regression model
# TODO: Make predictions and calculate RMSE and R²
""",
                    "solution": """
# Split the data
X_train, X_test, y_train, y_test = train_test_split(sizes, prices, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
""",
                    "explanation": "This creates a basic linear regression model and evaluates performance using RMSE and R²."
                }
            },
            
            "K-Means": {
                "Beginner": {
                    "problem": "Implement K-Means clustering to group customers based on their spending patterns.",
                    "starter_code": """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample customer data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# TODO: Apply K-Means clustering with k=4
# TODO: Plot the results with different colors for each cluster
# TODO: Display cluster centers
""",
                    "solution": """
# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Cluster centers:\\n{kmeans.cluster_centers_}")
""",
                    "explanation": "This implements K-Means clustering, visualizes results, and shows cluster centroids."
                }
            },
            
            "Logistic Regression": {
                "Beginner": {
                    "problem": "Build a logistic regression model to classify emails as spam or not spam.",
                    "starter_code": """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample data: email features and labels (0=not spam, 1=spam)
np.random.seed(42)
X = np.random.rand(1000, 5)  # 5 features
y = np.random.randint(0, 2, 1000)  # binary labels

# TODO: Split the data
# TODO: Create and train logistic regression model
# TODO: Make predictions and evaluate accuracy
""",
                    "solution": """
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\\nModel coefficients: {model.coef_[0]}")
""",
                    "explanation": "This creates a logistic regression classifier for binary classification and shows model coefficients."
                }
            },
            
            "Encoding": {
                "Beginner": {
                    "problem": "Implement different encoding techniques for categorical variables.",
                    "starter_code": """
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample dataset with categorical variables
data = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston'],
    'salary': [50000, 80000, 95000, 120000, 75000]
}
df = pd.DataFrame(data)

# TODO: Apply Label Encoding to 'education' (ordinal)
# TODO: Apply One-Hot Encoding to 'city' (nominal)
# TODO: Display the encoded dataset
""",
                    "solution": """
# Label Encoding for ordinal data (education has natural order)
education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
df['education_encoded'] = df['education'].map(education_mapping)

# One-Hot Encoding for nominal data (city has no natural order)
city_encoded = pd.get_dummies(df['city'], prefix='city')
df_encoded = pd.concat([df, city_encoded], axis=1)

print("Original dataset:")
print(df)
print("\\nEncoded dataset:")
print(df_encoded[['education_encoded', 'city_Boston', 'city_Chicago', 'city_LA', 'city_NYC', 'salary']])
""",
                    "explanation": "This demonstrates proper encoding techniques: Label Encoding for ordinal data and One-Hot Encoding for nominal data."
                }
            },
            
            "Feature Scaling": {
                "Beginner": {
                    "problem": "Apply different feature scaling techniques and compare their effects.",
                    "starter_code": """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample dataset with different scales
data = {
    'age': [25, 35, 45, 55, 65],
    'income': [30000, 50000, 80000, 120000, 150000],
    'score': [85, 92, 78, 95, 88]
}
df = pd.DataFrame(data)

# TODO: Apply StandardScaler
# TODO: Apply MinMaxScaler  
# TODO: Apply RobustScaler
# TODO: Compare the results
""",
                    "solution": """
# Apply different scalers
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
df_robust = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)

print("Original data:")
print(df)
print("\\nStandardScaler (mean=0, std=1):")
print(df_standard.round(3))
print("\\nMinMaxScaler (range 0-1):")
print(df_minmax.round(3))
print("\\nRobustScaler (median-based):")
print(df_robust.round(3))
""",
                    "explanation": "This compares different scaling techniques: StandardScaler for normal data, MinMaxScaler for bounded ranges, and RobustScaler for data with outliers."
                }
            },
            
            "Polynomial Regression": {
                "Beginner": {
                    "problem": "Create polynomial regression models with different degrees and compare performance.",
                    "starter_code": """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 1, 50).reshape(-1, 1)
y = 2 * X.ravel() + 3 * X.ravel()**2 + np.random.normal(0, 0.1, 50)

# TODO: Create polynomial regression models with degrees 1, 2, and 3
# TODO: Fit the models and make predictions
# TODO: Compare RMSE and R² scores
""",
                    "solution": """
degrees = [1, 2, 3]
results = {}

for degree in degrees:
    # Create polynomial pipeline
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit and predict
    poly_model.fit(X, y)
    y_pred = poly_model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    results[degree] = {'RMSE': rmse, 'R²': r2}
    print(f"Degree {degree}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

# Plot results
plt.figure(figsize=(12, 4))
for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i+1)
    poly_model = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression())])
    poly_model.fit(X, y)
    y_pred = poly_model.predict(X)
    plt.scatter(X, y, alpha=0.6)
    plt.plot(X, y_pred, 'r-', linewidth=2)
    plt.title(f'Degree {degree}')
plt.tight_layout()
plt.show()
""",
                    "explanation": "This demonstrates how polynomial degree affects model complexity and performance, showing the bias-variance trade-off."
                }
            },
            
            "K-NN": {
                "Beginner": {
                    "problem": "Implement K-NN classifier and find the optimal K value using cross-validation.",
                    "starter_code": """
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_classification(n_samples=500, n_features=4, n_classes=3, random_state=42)

# TODO: Split the data and scale features
# TODO: Test different K values (1, 3, 5, 7, 9)
# TODO: Use cross-validation to find optimal K
# TODO: Train final model with best K
""",
                    "solution": """
# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different K values
k_values = [1, 3, 5, 7, 9]
cv_scores = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores[k] = scores.mean()
    print(f"K={k}: CV Score = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Find best K
best_k = max(cv_scores, key=cv_scores.get)
print(f"\\nBest K: {best_k}")

# Train final model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
test_score = final_knn.score(X_test_scaled, y_test)
print(f"Test accuracy with K={best_k}: {test_score:.4f}")
""",
                    "explanation": "This shows how to find optimal K using cross-validation and demonstrates the importance of feature scaling for K-NN."
                }
            },
            
            "Hierarchical Clustering": {
                "Beginner": {
                    "problem": "Perform hierarchical clustering and visualize the dendrogram.",
                    "starter_code": """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate sample data
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

# TODO: Perform hierarchical clustering with different linkage methods
# TODO: Create and plot dendrogram
# TODO: Extract clusters and visualize results
""",
                    "solution": """
# Perform hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average']

plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods):
    # Compute linkage matrix
    linkage_matrix = linkage(X, method=method)
    
    # Plot dendrogram
    plt.subplot(2, 3, i+1)
    dendrogram(linkage_matrix, truncate_mode='level', p=3)
    plt.title(f'Dendrogram ({method} linkage)')
    
    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
    cluster_labels = clustering.fit_predict(X)
    
    # Plot clusters
    plt.subplot(2, 3, i+4)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    plt.title(f'Clusters ({method} linkage)')

plt.tight_layout()
plt.show()

print("Hierarchical clustering completed with different linkage methods")
print("Ward linkage minimizes within-cluster variance")
print("Complete linkage uses maximum distance between clusters")
print("Average linkage uses average distance between all pairs")
""",
                    "explanation": "This demonstrates hierarchical clustering with different linkage methods and shows how dendrograms help visualize cluster formation."
                }
            },
            
            "PCA": {
                "Beginner": {
                    "problem": "Apply PCA for dimensionality reduction and analyze explained variance.",
                    "starter_code": """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# TODO: Standardize the features
# TODO: Apply PCA and analyze explained variance
# TODO: Visualize data in 2D using first two components
# TODO: Determine number of components for 95% variance
""",
                    "solution": """
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Analyze explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\\nCumulative explained variance: {cumulative_variance}")

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Visualize in 2D
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
plt.title('PCA - First Two Components')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(explained_variance)+1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.legend()

plt.tight_layout()
plt.show()
""",
                    "explanation": "This demonstrates PCA for dimensionality reduction, showing how to analyze explained variance and choose the optimal number of components."
                }
            },
            
            "SVM": {
                "Beginner": {
                    "problem": "Build SVM classifiers with different kernels and compare performance.",
                    "starter_code": """
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=4, n_classes=2, random_state=42)

# TODO: Split and scale the data
# TODO: Train SVM with different kernels (linear, rbf, poly)
# TODO: Compare performance of different kernels
# TODO: Analyze support vectors
""",
                    "solution": """
# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different kernels
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    # Train SVM
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[kernel] = {
        'accuracy': accuracy,
        'n_support_vectors': svm.n_support_,
        'support_vectors_total': len(svm.support_)
    }
    
    print(f"\\n{kernel.upper()} Kernel:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Support vectors per class: {svm.n_support_}")
    print(f"Total support vectors: {len(svm.support_)}")

# Find best kernel
best_kernel = max(results, key=lambda k: results[k]['accuracy'])
print(f"\\nBest performing kernel: {best_kernel}")
print(f"Best accuracy: {results[best_kernel]['accuracy']:.4f}")
""",
                    "explanation": "This compares SVM performance with different kernels and shows how support vectors vary across kernel types."
                }
            },
            
            "Cross Validation": {
                "Beginner": {
                    "problem": "Implement different cross-validation strategies and compare model performance.",
                    "starter_code": """
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

# TODO: Implement different CV strategies (KFold, StratifiedKFold, LeaveOneOut)
# TODO: Compare two different models using CV
# TODO: Analyze the variance in CV scores
""",
                    "solution": """
# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Define CV strategies
cv_strategies = {
    '5-Fold CV': KFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Leave-One-Out': LeaveOneOut()
}

# Compare models with different CV strategies
for cv_name, cv_strategy in cv_strategies.items():
    print(f"\\n{cv_name}:")
    print("-" * 40)
    
    for model_name, model in models.items():
        if cv_name == 'Leave-One-Out' and len(X) > 100:
            print(f"{model_name}: Skipped (too computationally expensive)")
            continue
            
        scores = cross_val_score(model, X, y, cv=cv_strategy)
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"{model_name}:")
        print(f"  Mean CV Score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print(f"  Individual scores: {scores}")

# Detailed analysis with 5-fold CV
print("\\n" + "="*50)
print("DETAILED 5-FOLD CV ANALYSIS")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"\\n{model_name}:")
    print(f"Fold scores: {[f'{score:.4f}' for score in scores]}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    print(f"95% CI: [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]")
""",
                    "explanation": "This demonstrates different cross-validation strategies and shows how to properly evaluate and compare model performance with confidence intervals."
                }
            }
        }

    def generate_coding_problem(self, topic: str, difficulty: str = "Beginner") -> Dict[str, Any]:
        """Generate a coding problem for the given topic"""
        if topic in self.coding_templates and difficulty in self.coding_templates[topic]:
            return self.coding_templates[topic][difficulty]
        
        # Fallback for topics not yet implemented
        return {
            "problem": f"Practice problem for {topic} is being developed. Try these available topics: {', '.join(self.coding_templates.keys())}",
            "starter_code": "# Coming soon...",
            "solution": "# Solution will be provided soon...",
            "explanation": "This topic's coding problems are under development.",
            "problem_type": "general",
            "test_data": {}
        }
    
    def get_quiz_questions(self, topic: str, num_questions: int = 5) -> List[Dict]:
        """Get quiz questions for a specific topic"""
        if topic not in self.problem_templates:
            return [{
                "question": f"Quiz questions for {topic} are being developed. Try these available topics: {', '.join(self.problem_templates.keys())}",
                "options": ["Coming soon", "Under development", "Check back later", "All of the above"],
                "correct": 3,
                "explanation": "This topic's quiz questions are under development."
            }]
        
        problems = self.problem_templates[topic]["problems"]
        # Return all available questions (up to num_questions)
        return problems[:min(num_questions, len(problems))]
    
    def generate_dataset_problem(self, topic: str) -> Dict[str, Any]:
        """Generate a dataset-based problem for hands-on practice"""
        dataset_problems = {
            "Linear Regression": {
                "description": "Predict house prices based on multiple features",
                "data_generator": lambda: make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42),
                "target_name": "price",
                "feature_names": ["size", "bedrooms", "location_score"]
            },
            
            "Classification": {
                "description": "Classify customers into different segments",
                "data_generator": lambda: make_classification(n_samples=200, n_features=4, n_classes=3, random_state=42),
                "target_name": "customer_segment",
                "feature_names": ["age", "income", "spending_score", "loyalty_years"]
            },
            
            "K-Means": {
                "description": "Cluster customers based on purchasing behavior",
                "data_generator": lambda: make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42),
                "target_name": "cluster",
                "feature_names": ["annual_spending", "frequency_of_purchase"]
            }
        }
        
        if topic in dataset_problems:
            problem_info = dataset_problems[topic]
            X, y = problem_info["data_generator"]()
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=problem_info["feature_names"])
            df[problem_info["target_name"]] = y
            
            return {
                "description": problem_info["description"],
                "dataset": df,
                "features": problem_info["feature_names"],
                "target": problem_info["target_name"],
                "task_type": "regression" if "Regression" in topic else "classification" if "Classification" in topic else "clustering"
            }
        
        return {
            "description": f"Dataset problem for {topic} is not available yet.",
            "dataset": pd.DataFrame(),
            "features": [],
            "target": "",
            "task_type": "unknown"
        }
    
    def get_problem_difficulty_levels(self) -> List[str]:
        """Get available difficulty levels"""
        return self.difficulty_levels
    
    def get_available_topics(self) -> List[str]:
        """Get list of topics with available problems"""
        return list(self.problem_templates.keys())
    
    def generate_comprehensive_exercise(self, topic: str, difficulty: str = "Beginner") -> Dict[str, Any]:
        """Generate a comprehensive exercise including quiz, coding, and dataset problems"""
        return {
            "topic": topic,
            "difficulty": difficulty,
            "quiz_questions": self.get_quiz_questions(topic, 2),
            "coding_problem": self.generate_coding_problem(topic, difficulty),
            "dataset_problem": self.generate_dataset_problem(topic),
            "learning_objectives": self._get_learning_objectives(topic)
        }
    
    def _get_learning_objectives(self, topic: str) -> List[str]:
        """Get learning objectives for a topic"""
        objectives = {
            "Linear Regression": [
                "Understand the linear relationship assumption",
                "Learn to evaluate model performance using RMSE",
                "Practice feature scaling and data preprocessing"
            ],
            "K-Means": [
                "Understand centroid-based clustering",
                "Learn to choose optimal number of clusters",
                "Practice data visualization and interpretation"
            ],
            "Logistic Regression": [
                "Understand probability-based classification",
                "Learn about sigmoid function and decision boundaries",
                "Practice binary classification evaluation metrics"
            ]
        }
        
        return objectives.get(topic, [f"Master the fundamentals of {topic}"])
