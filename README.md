# CSE150A_Project: Game Recommendation System
### By: Derrick Nguyen, Amelia Lei, Jahnavi Naik

## PEAS/Agent Analysis:

### Describe your agent in terms of PEAS and give a background of your task at hand.

Our AI agent is a game recommendation agent that aims to recommend a game to the user based on their preferences and recommendations of other games they have played in the past. In terms of PEAS, we have:

- Performance Measure: the feedback that the user gives from the games that have been recommended to them, and how satisfied they are with it.
- Environment: the user profiles and their preferences, along with all of the game data.
- Actuators: the games that the agent recommends to the user based on the preferences.
- Sensors: the information about the game and users that the agent uses to generate personalized recommendations. This includes the game catalog such as genre and ratings and the user preferences such as previous games they’ve enjoyed, favorite genres, or their budget.


## Agent Setup, Data Preprocessing, Training setup

### Give an exploration of your dataset, and highlight which variables are important. Give a brief overview of each variable and its role in your agent/model.

For our project, we used 2 datasets. One includes many steam games along with different attributes that describe them such as price, genre, release date, required age, about the game, etc. We also have a dataframe that has user information, which tells us games that different users have played, if they recommend them, and their reviews. After pre-processing these dataframes, we found that the important variables we would use are review ratio, is free, genres, price tier, and playtime tier. These are variables that we engineered using the available features. Review ratio is a ratio of positive over negative reviews to determine the overall sentiment of the game. Is free tells us if the game is free to play or not. Price tier places the games into categories of prices, and playtime tier gives us categories of playtimes.

![bayes_net](https://github.com/user-attachments/assets/0808ef3e-0626-47df-a597-69eca1403c5a)


### Describe in detail how your variables interact with each other, and if your model fits a particular structure, explain why you chose that structure to model your agent. If it does not, further elaborate on why you chose that model based on the variables.

Here is how each of the features interact with the target variable (is_recommended) and with each other:
- review_ratio:
  - review_ratio - is_recommended: Games with a higher review ratio are more likely to be recommended 
  - review_ratio - playtime_tier: If a game has higher playtime, it is more likely to have positive reviews because players are enjoying the game.
- price_tier: 
  - price_tier - is_recommended: Cheaper games are more likely to be recommended because they are more accessible 
  - price_tier - review_ratio: players tend to have higher expectations for more expensive games to make sure it’s worth their money which can result in lower review scores for more expensive games 
- playtime_tier:
  - playtime_tier - is_recommended: games with a higher playtime are more likely to be recommended because they’re more engaging 
  - playtime_tier - genre: certain games tend to have higher playtimes such as strategy games and MMOs
  - playtime_tier - review_ratio: games with higher playtimes tend to have more engaged players which leads to higher reviews
- genre:
  - genre - is_recommneded: Certain genres (e.g. action and RPG) tend to have higher recommendation rates regardless of other features 
  - genre - price: genres like AAA action-adventure with longer storylines and better graphics are often more expensive compared to casual indie games 
  - genre - playtime: some genres have naturally higher playtimes than others due to their storylines and quests like MMORGs

We chose a Naive Bayes model for this structure because even though these interactions exist, Naive Bayes assumes feature independence. This means that the model is calculating:

P(is_recommended∣X) = P(genre∣is_recommended)P(price_tier∣is_recommended)P(playtime_tier∣is_recommended)P(review_ratio∣is_recommended)P(is_recommended)  

Therefore, even if price_tier and review_ratio influence each other, the model treats them separately. This simplifies the model and makes it more efficient. However, one drawback is that it might miss more complex dependencies such as expensive games having lower review scores. 


### Describe your process for calculating parameters in your model. That is, if you wish to find the CPTs, provide formulas as to how you computed them. If you used algorithms in class, just mention them.

For discrete variables such as price tier, playtime tier, and genre, we used Maximum Likelihood Estimation (MLE) to compute the probability of each category occurring given the recommendation status. This was done by counting the occurrences of each feature value within each class and dividing it by the total number of instances in that class. For a categorical feature $X_i$ the probability of a specific value given the class label Y is calculated as:

$P(X_i = x | Y = y) = count(X_i = x, Y = y) / count(Y = y)$

For continuous features like the review ratio, we assumed a Gaussian distribution and calculated the parameters (mean and variance) for each class. This involved computing the mean $u_y,i$ and variance $σ²_y,i$ of the feature within each class using the following formulas:

$μ_y,i = (1 / N_y) * Σ (x_j,i)$

$σ²_y,i = (1 / N_y) * Σ (x_j,i - μ_y,i)²$

where $N_y$  is the number of instances belonging to class Y = y and $x_j,i$ represents individual feature values. The probability density function (PDF) for a given value $x_i$ was then computed using the Gaussian formula:

$P(x_i | y) = (1 / sqrt(2πσ²_y,i)) * exp(- (x_i - μ_y,i)² / 2σ²_y,i)$

After calculating these parameters, we used Bayes' Theorem to determine the posterior probability of a game being recommended given its feature values.

### Important: if you used a library like pgmpy, etc., be sure to explain what it does. Then, cite your source (library homepage link or documentation is sufficient) at the bottom of your README.md file. If you use a particular algorithm/model/structure not covered in this class as your core structure (i.e. GaussianHMM, etc.) you must carefully explain the formulation behind this structure. How does it differ from its discrete analog (if it has one?) How do you perform inference on it?

### If you use a particular algorithm/model/structure not covered in this class as a non-core structure (i.e. RandomHillCimb to create BN dependencies) you may briefly explain what the function does. Remember to provide a source.
Gaussian Naive Bayes differ from regular naive bayes in the way that it handles the model's feature distributions. This is because Gaussian Naive Bayes handles continuous features by assuming they follow a Normal distribution. Instead of using feature frequencies, it computes the probability of each feature given a class using the Gaussian probability density function (PDF), which depends on the mean and variance of the feature per class. The inference process in both models follows Bayes' Theorem, but in GNB, P(Xi|Y) is derived from the Gaussian PDF rather than categorical probabilities. 
  
$P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma^2_{y,i}}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma^2_{y,i}}\right)$
  
GaussianNB is well-suited for numerical data like game prices or user ratings as well as our review ratio variable (a continuous variable), whereas regular Naïve Bayes is more effective for text-based applications. We implement Gaussian Naive Bayes using the model function from the scikit-learn library, where the .fit() function estimates the mean and variance for each feature per class, and .predict() computes posteriors using the Gaussian PDF to classify new data points.

Gaussian Naive Bayes Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html 

## Train Your Model

### Be sure to link to a clean, documented portion of code in your notebook or provide a code snippet in the README.

Our preprocessing, data exploration, and training can be found in this notebook (Agent #2): [View the Notebook](proj.ipynb)

### How did you train your Gaussian Naive Bayes?
 
 We trained our Gaussian Naive Bayes by analyzing the dataset and turning them into meaningful states. For our training we:
 
 Preprocessed the Data:
 
The dataset was processed by engineering new features, such as review ratio (calculated as the ratio of positive to negative reviews) and categorical encoding of price and playtime tiers using binning. Additionally, genres were one-hot encoded to represent different game categories.
 
 Trained the Gaussian Naive Bayes:
 
We selected relevant features, including review ratio, genre encodings, price tier, and playtime tier, and applied label encoding to categorical variables. The dataset was split into training (80%) and testing (20%) sets. We then trained a Gaussian Naive Bayes (GNB) model, assuming that continuous variables follow a Gaussian distribution and categorical variables are treated independently using MLE-based probability estimation. The model was fit using scikit-learn’s GaussianNB implementation.
 
 Evaluating Model Performance:
 
We evaluated the model using accuracy. Predictions were generated on the test set, and a confusion matrix was plotted to analyze classification performance. Additionally, we implemented a recommendation function that inferred user preferences from their history and used the trained GNB model to predict recommendations for unplayed games. These recommendations were ranked based on the predicted probability of recommendation.
 
 
 ### Can you explain what Gaussian Naive Bayes is?
 
Gaussian Naive Bayes (GNB) is a classification algorithm based on Bayes’ Theorem, which assumes that features are conditionally independent given the class label. The Gaussian variant of Naive Bayes assumes that continuous features follow a normal (Gaussian) distribution, making it suitable for datasets with numerical attributes like review ratios, playtime tiers, and price tiers.
 
 
 ### How do its training algorithms work?
 
Training a Gaussian Naive Bayes classifier involves the following steps:

 - Calculate Class Priors: Compute the probability of each class (Recommended vs. Not Recommended) based on their frequency in the training data.
 - Estimate Gaussian Parameters: For each numerical feature (review ratio, playtime tier, price tier), compute the mean (μ) and variance (σ²) separately for each class. This assumes that the feature values within each class are normally distributed.
 - Apply Bayes' Theorem: When making predictions, the model uses the probability density function (PDF) of a Gaussian distribution to calculate the likelihood of a feature belonging to a particular class
 - Compute Posterior Probability: The posterior probability for each class is computed using Bayes’ Theorem, and the class with the highest probability is chosen as the prediction.
 
The model iterates over these steps during training, learning the parameters μ and σ² for each feature-class combination, and then uses these parameters to classify new data points.

## Conclusion/Results

### Describe in detail your results, including any helpful visualizations like heatmaps, confusion matrices, etc. (if applicable). Please provide numerical results (unless your model's performance metrics do not include numbers).
![confusion_matrix](https://github.com/user-attachments/assets/d8a1100c-8d30-4f3c-979f-f59b67accb7d)

To test the accuracy of our model, we split the data into X, which contained all of our features, and y, which contained the is_recommended column. We then use the train test split to split it into training and testing data. We then used the accuracy_score function from the sklearn metrics library to get our accuracy. This works because our is_recommended column tells us games that the user has previously played and either recommends or not, so we test the model to see if it can correctly guess these past games as a game the user would play or not. Our model scores an accuracy of 78.66%. 

#### Fitting Graph

![image](https://github.com/user-attachments/assets/56c342dd-50de-4609-aff5-14b7bb2ddf5d)

We also created a fitting graph for our model to test whether it was overfititng or underfitting to our data. From our fitting graph, we can see that the testing and training accuracies are very close to each other no amtter the number of training samples. Since the accuracoes are almost identical, this means that our model does not have signs of overfitting or underfitting and generalizes well to our data. These results make sense given the nature of a naive bayes model. Since the model assumes conditional independance for all features given the class label, this simplifies the model and reduces the risk of overfitting as it doesn't learn any complex relationships. The probabalistic process of this mdoel also makes it more robust to noise as it's learning the likelihood of the data instead of specific patterns. 

### Be sure to interpret your results! If you obtain a poor performance, compare it to some baseline easy task (i.e. random guessing, etc.) so you can have an estimate as to where your model performance is at. Propose various points of improvement for your model, and be thorough! Do not just say "we could use more data", or "I'll be sure to use reinforcement learning for the next milestone so we can better model our objective." Carefully work through your data preprocessing, your training steps, and point out any simplifications that may have impacted model performance, or perhaps any potential errors or biases in the dataset. You are not required to implement these points of improvement unless it is clear that your original model is significantly lacking in detail or effort.

Interpretation of Results:

Our Gaussian Naive Bayes (GNB) model achieved an accuracy of 0.7875, meaning it correctly classified approximately 79% of user recommendations based on the available features.

Points of Improvement:

- Handling Outliers: Gaussian Naive Bayes assumes normally distributed features, but real-world data often contains skewness and outliers. Applying log transformations or scaling methods (e.g., standardization) could improve the distribution of features and enhance performance.
- Class Conditional Probability Smoothing: If some games have very few recommendations, their estimated probabilities could be unreliable. Laplace smoothing or alternative regularization techniques could prevent probabilities from collapsing to zero
- Tree-Based Models: These models can capture complex feature interactions better than Naive Bayes and might improve accuracy.

## Citations:

We used ChatGPT to help us understand Gaussian Naive Bayes and how to implement it with our dataset.
