Name    = "Ehtisham Ahmad"

email   = "ehtishamahmed10@gmail.com"


# Machine Learning Basics

## Types of Machine Learning

1. Supervised Learning
2. Unsupervised Learning
3. Semi-supervised Learning
4. Reinforcement Learning
* 
# Types and Algorithms explanation

## 1. Supervised Learning:

   - Work under supervision
   - Teacher teaches
   - Predicition
   - Outcome

### Types:
- Classification (for categeories)
- Regression (for numerical data)
  
### Algorithms:

- Logistic Regression:

Based on probablity concept and logic calculation, the hypothesis of logistic regression tends it to limit the cost function between 0 and 1.

- K-Nearest Neighbors (K-NN):

K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

- Support Vector Machines(SVM):

Support vector machines are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. Support Vectors are simply the coordinates of individual observation.

- Kernel SVM:

Kernel Function is a method used to take data as input and transform into the required form of processing data. “Kernel” is used due to set of mathematical functions used in Support Vector Machine provides the window to manipulate the data.

- Naive bayes:

It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

- Decision Tree classification:

Decision Trees are a non-parametric supervised learning method used for both classification and regression tasks. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

- Random forest classification:

It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.

## 2. Unsupervised/Clustring Learning:
-   No supervision
-   No teacher
-   Self learning
-   No labelling of data
-   Find patterns by itself

### Algorithms:

- K-Means Clustering:

Identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

- Hierarchical Clustering:

Another unsupervised machine learning approach for grouping unlabeled datasets into clusters. The hierarchy of clusters is developed in the form of a tree in this technique, and this tree-shaped structure is known as the dendrogram.

- Probabilistic Clustering:

 A classifier that is able to predict, given an observation of an input, a probability distribution over a set of classes, rather than only outputting the most likely class that the observation should belong to.

 ## 3. Semi-Supervised Learning:
-   Mixture of Supervised and Unsupervised learning.
-   Some data is labelles, most is not.
-   Some is input(supervised) data and some is clustered(unsupervised) data.

## 4. Reinforcement Learning:
-   Hit and trial learning.
-   Learn from mistakes.
-   Reward and punishment rule.
-   Prediction based on reward and punishment.
-   Depends on feedback.

### Algorithms:

1. Model-Free Reinforcement learning:

Model-free methods primarily rely on learning.

- Policy Optimization:

This method views reinforcement learning as a nu- merical optimization problem where we optimize the expected reward with respect to the policy's parameters.

- Q - Learning:
Q-learning is a model-free reinforcement learning algorithm. Q-learning is a values-based learning algorithm. Value based algorithms updates the value function based on an equation.

1. Model-Based Reinforcement learning:

Model-based methods rely on planning as their primary component

- Learn the model.
- Given the model.