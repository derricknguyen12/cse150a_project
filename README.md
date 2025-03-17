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

For our project, we used 2 datasets. One includes many steam games along with different attributes that describe them such as price, genre, release date, required age, about the game, etc. We also have a dataframe that has user information, which tells us games that different users have played, if they recommend them, and their reviews. After pre-processing these dataframes, we found that the important variables we would use are review ratio, genres, price tier, and playtime tier. These are variables that we engineered using the available features. 
- review_tier: Review tier represents the proportion of positive reviews a game has received classified into different ranges ('Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'). It helps us determine the overall sentiment of the game. 
-  genres: Each game is encoded into multiple genre categories (e.g., Action, RPG, Strategy). This allows the model to identify which genres a user frequently engages with and recommend games within those preferred genres. We used binary encoding, where each genre is represented as a separate feature (genre_action, genre_rpg)
-  price_tier: Games can be classified into different prices ranges ('Free', 'Budget', 'Mid', 'Premium', 'Deluxe'). This allows our model to identify whether users prefer premium or affordable games. This feature helps get more accurate recommendations based on affordability preferences.
-  playtime_tier: Playtime tier categorizes the total hours spent on a game into different ranges ('Very Low', 'Low', 'Medium', 'High', 'Very High'). Users who play long-playtime games may prefer similar recommendations, whereas casual players may prefer shorter games. This helps our model capture general player behavior trends.

![bayes_net](https://github.com/user-attachments/assets/ea6c4565-068f-4d95-867f-1b86a50b24f3)



### Describe in detail how your variables interact with each other, and if your model fits a particular structure, explain why you chose that structure to model your agent. If it does not, further elaborate on why you chose that model based on the variables.

Here is how each of the features interact with the target variable (is_recommended) and with each other:
- review_tier:
  - review_tier - is_recommended: Games with a higher review ratio are more likely to be recommended 
  - review_tier - playtime_tier: If a game has higher playtime, it is more likely to have positive reviews because players are enjoying the game.
- price_tier: 
  - price_tier - is_recommended: Cheaper games are more likely to be recommended because they are more accessible 
  - price_tier - review_tier: players tend to have higher expectations for more expensive games to make sure it’s worth their money which can result in lower review scores for more expensive games 
- playtime_tier:
  - playtime_tier - is_recommended: games with a higher playtime are more likely to be recommended because they’re more engaging 
  - playtime_tier - genre: certain games tend to have higher playtimes such as strategy games and MMOs
  - playtime_tier - review_tier: games with higher playtimes tend to have more engaged players which leads to higher reviews
- genre:
  - genre - is_recommneded: Certain genres (e.g. action and RPG) tend to have higher recommendation rates regardless of other features 
  - genre - price: genres like AAA action-adventure with longer storylines and better graphics are often more expensive compared to casual indie games 
  - genre - playtime: some genres have naturally higher playtimes than others due to their storylines and quests like MMORGs

We chose a Naive Bayes model for this structure because even though these interactions exist, Naive Bayes assumes feature independence. This means that the model is calculating:

P(is_recommended | X) = P(genre | is_recommended) P(price_tier | is_recommended) P(playtime_tier | is_recommended) P(review_tier | is_recommended) P(is_recommended)

Therefore, even if price_tier and review_tier influence each other, the model treats them separately. This simplifies the model and makes it more efficient. However, one drawback is that it might miss more complex dependencies such as expensive games having lower review scores. 


### Describe your process for calculating parameters in your model. That is, if you wish to find the CPTs, provide formulas as to how you computed them. If you used algorithms in class, just mention them.

Since we made our variables (price_tier, playtime_tier, review_tier, genre) discrete variables by binning them, we used Maximum Likelihood Estimation (MLE) to compute the probability of each category occurring given the recommendation status. This was done by counting the occurrences of each feature value within each class and dividing it by the total number of instances in that class. 
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>price_tier</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <th>is_recommended</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>0.327178</td>
      <td>0.066105</td>
      <td>0.164680</td>
      <td>0.254967</td>
      <td>0.187068</td>
      <td>9.635053e-07</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.315368</td>
      <td>0.059344</td>
      <td>0.112527</td>
      <td>0.316605</td>
      <td>0.196153</td>
      <td>3.285415e-06</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>playtime_tier</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>is_recommended</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>0.056595</td>
      <td>0.208426</td>
      <td>0.048009</td>
      <td>0.012659</td>
      <td>0.674310</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.094705</td>
      <td>0.354232</td>
      <td>0.091120</td>
      <td>0.015954</td>
      <td>0.443988</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>review_tier</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>is_recommended</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>0.018682</td>
      <td>0.024609</td>
      <td>0.052432</td>
      <td>0.001968</td>
      <td>0.902308</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.001728</td>
      <td>0.003478</td>
      <td>0.011047</td>
      <td>0.000429</td>
      <td>0.983318</td>
    </tr>
  </tbody>
</table>
</div>

For a categorical feature $X_i$ the probability of a specific value given the class label Y is calculated as:

$P(X_i = x | Y = y) = count(X_i = x, Y = y) / count(Y = y)$

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_Movie</th>
      <th>genre_Early Access</th>
      <th>genre_Indie</th>
      <th>genre_Casual</th>
      <th>genre_Adventure</th>
      <th>genre_Racing</th>
      <th>genre_Strategy</th>
      <th>genre_Photo Editing</th>
      <th>genre_RPG</th>
      <th>genre_Game Development</th>
      <th>...</th>
      <th>genre_Sports</th>
      <th>genre_Accounting</th>
      <th>genre_Web Publishing</th>
      <th>genre_Audio Production</th>
      <th>genre_Simulation</th>
      <th>genre_Gore</th>
      <th>genre_Sexual Content</th>
      <th>genre_Software Training</th>
      <th>genre_Animation &amp; Modeling</th>
      <th>genre_Utilities</th>
    </tr>
    <tr>
      <th>is_recommended</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>0.000071</td>
      <td>0.022426</td>
      <td>0.177650</td>
      <td>0.065329</td>
      <td>0.156400</td>
      <td>0.010853</td>
      <td>0.070844</td>
      <td>0.000031</td>
      <td>0.095216</td>
      <td>0.000099</td>
      <td>...</td>
      <td>0.009982</td>
      <td>3.565159e-06</td>
      <td>0.000153</td>
      <td>0.000046</td>
      <td>0.069569</td>
      <td>0.001012</td>
      <td>0.000179</td>
      <td>0.000126</td>
      <td>0.00023</td>
      <td>0.000520</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.000020</td>
      <td>0.018494</td>
      <td>0.198295</td>
      <td>0.069984</td>
      <td>0.171629</td>
      <td>0.009550</td>
      <td>0.067834</td>
      <td>0.000043</td>
      <td>0.091273</td>
      <td>0.000084</td>
      <td>...</td>
      <td>0.007871</td>
      <td>7.022349e-07</td>
      <td>0.000182</td>
      <td>0.000044</td>
      <td>0.068070</td>
      <td>0.000633</td>
      <td>0.000165</td>
      <td>0.000067</td>
      <td>0.00020</td>
      <td>0.000474</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>

To calculate P(is_recommended) which represents the probability that a game was recommended (True) or not (False), we used the following formulas:

P(is_recommended = True) = Number of Recommended Games / Total Number of Games

P(is_recommended = False) = Number of Not Recommended Games / Total Number of Games

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>is_recommended</th>
      <th>proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True</th>
      <td>0.814781</td>
    </tr>
    <tr>
      <th>False</th>
      <td>0.185219</td>
    </tr>
  </tbody>
</table>
</div>

After calculating these parameters, we used Bayes' Theorem to determine the posterior probability of a game being recommended given its feature values.



### Important: if you used a library like pgmpy, etc., be sure to explain what it does. Then, cite your source (library homepage link or documentation is sufficient) at the bottom of your README.md file. If you use a particular algorithm/model/structure not covered in this class as your core structure (i.e. GaussianHMM, etc.) you must carefully explain the formulation behind this structure. How does it differ from its discrete analog (if it has one?) How do you perform inference on it?

### If you use a particular algorithm/model/structure not covered in this class as a non-core structure (i.e. RandomHillCimb to create BN dependencies) you may briefly explain what the function does. Remember to provide a source.

 ### What is Categorical Naive Bayes?
Categorical Naive Bayes differ from regular naive bayes in the way that it handles the model's feature distributions. While other forms of naive bayes assumes features follow specific probability distributions such as the multinomial distribution for text data or the Gaussian distribution for continuous data, categorical naive bayes is specifically designed for discrete categorical features. This works well with our data because we binned all of our numerical variables, making them categorical variables. We also have categorical features such as genre which is appropriate for this model. Categorical Naive Bayes estimates the conditional probability of each feature given a class using frequency counts and applies Laplace smoothing to handle unseen category values. 

Categorical Naive Bayes Reference: [https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html ](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)

 ### How do its training algorithms work?
 
Training a Categorical Naive Bayes classifier involves the following steps:

 - Calculate Class Priors: Compute the probability of each class (Recommended vs. Not Recommended) based on their frequency in the training data.
 - Estimate Conditional Probabilities: For each categorical feature, the model calculates the probability of each category occurring within each class. These probabilities are estimated using frequency counts and are smoothed using Laplace smoothing to handle unseen category values.
 - Apply Bayes' Theorem: When making predictions, the model multiplies the prior probability of the class by the conditional probabilities of the observed feature values given that class.
 - Compute Posterior Probability: The posterior probability for each class is computed using Bayes’ Theorem, and the class with the highest probability is chosen as the prediction.
 
The model iterates over these steps during training, learning the probability distributions for each feature class combination and using them to classify new data points.

## Train Your Model

### Be sure to link to a clean, documented portion of code in your notebook or provide a code snippet in the README.

A more detailed version of our preprocessing, data exploration, and training can be found in this notebook (Agent #2): [View the Notebook](proj.ipynb)

Here is a snippet of what we did:
```
def preprocess_data(df):
    # calculate sentiment
    df['review_ratio'] = df['Positive'] / (df['Negative'] + 1)  

    # Bin review_ratio 
    df['review_tier'] = pd.cut(df['review_ratio'], 
                               bins=[-np.inf, 0.2, 0.5, 0.8, 1.2, np.inf], 
                               labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])

    # get all game genres
    genres = []
    for genre_list in df['Genres']:
        genres.extend(genre_list.split(','))
    unique_genres = list(set(genres))
    
    #encode genres
    for genre in unique_genres:
        df[f'genre_{genre}'] = df['Genres'].apply(lambda x: 1 if genre in x else 0)
    
    # bin prices
    df['price_tier'] = pd.cut(df['Price'], 
                             bins=[-0.01, 0.01, 10, 20, 40, 100], 
                             labels=['Free', 'Budget', 'Mid', 'Premium', 'Deluxe'])
    
    # bin playtimes hours
    df['playtime_tier'] = pd.cut(df['hours'], 
                               bins=[-0.01, 10, 50, 100, 500, 5000], 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    
    return df

# Preprocess Data
processed_df = preprocess_data(merged_df)

encoders = {}  
for col in ['review_tier', 'price_tier', 'playtime_tier']:
    encoders[col] = LabelEncoder()  
    processed_df[col] = encoders[col].fit_transform(processed_df[col])

# Model Features
features = ['review_tier', 'price_tier', 'playtime_tier'] + \
           [col for col in processed_df.columns if col.startswith('genre_')] 

X = processed_df[features]
y = processed_df['is_recommended']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes Model
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)

```


### How did you train your Categorical Naive Bayes?
 
We trained our Categorical Naive Bayes model by analyzing the dataset and converting the continuous and numerical variables into discrete categories to be used in our model. For our training we:
 
 Preprocessed the Data:
 
The dataset was processed by first engineering new features, such as review ratio (calculated as the ratio of positive to negative reviews) and binning of of price_tiers, playtime tiers, and review_tiers. Additionally, genres were encoded to represent different game categories.
 
 Trained the Categorical Naive Bayes:
 
We selected relevant features, including review tier, genre encodings, price tier, and playtime tier. Since Categorical Naïve Bayes (CNB) is designed for discrete categorical data, we label encoded price tier, playtime tier, and review tier as they were originally stored at strings. The binary genre encodings was kept as is. After, the dataset was split into training (80%) and testing (20%) sets. We then trained a Categorical Naïve Bayes model using scikit-learn’s CategoricalNB implementation which estimates probabilities based on observed categorical frequencies. 
 
 Evaluating Model Performance:
 
We evaluated the model using accuracy. Predictions were generated on the test set after being trained on the training set. Accuracy was calculated using accuracy_score(y_test, y_pred), representing the proportion of correctly classified instances. We also plotted a confusion matrix to analyze classification performance by displaying the number of true positives (correctly recommended games), true negatives (correctly not recommended games), false positives (incorrectly recommended games), and false negatives (incorrectly not recommended games). Additionally, we implemented a recommendation function that inferred user preferences from their history and used the trained CNB model to predict recommendations for unplayed games. Using these inferred preferences, we generated recommendations by identifying unplayed games and prediciting their likelihood of being recommended. These recommendations were ranked based on the predicted probability of recommendation.


## Conclusion/Results

### Describe in detail your results, including any helpful visualizations like heatmaps, confusion matrices, etc. (if applicable). Please provide numerical results (unless your model's performance metrics do not include numbers).
![confusion_matrix](https://github.com/user-attachments/assets/b0cf396d-2423-406f-99f8-d7307a88ad1c)


To test the accuracy of our model, we split the data into X, which contained all of our features, and y, which contained the is_recommended column. We then use the train test split to split it into training and testing data. We then used the accuracy_score function from the sklearn metrics library to get our accuracy. This works because our is_recommended column tells us games that the user has previously played and either recommends or not, so we test the model to see if it can correctly guess these past games as a game the user would play or not. Our model scores an accuracy of 81.82%. 

The above confusion matrix tells us the counts for how many true positives, false positives, true negatives, and false negatives our model predicted. The high number of false positives (182,151) suggests that the model is biased towards recommending games. This might be because of the assumption of independence among features. Moreover, the number of false negatives (21,538) suggests that some good recommendations were missed. However, the dominant true positive count (891,411) shows that the model is still effective at making correct recommendations most of the time. According to the confusion matrix, our model has a precision rate of 83.03%, and a recall of 97.64%. This shows that our model is good at recommending games, as the precision, recall, and accuracy are very high.

#### Fitting Graph

![fitting_graph](https://github.com/user-attachments/assets/e97c55c3-8d2e-4d0e-95fd-6e3fee553787)


We also created a fitting graph for our model to test whether it was overfititng or underfitting to our data. From our fitting graph, we can see that the testing and training accuracies are very close to each other no matter the number of training samples. Since the accuracies are almost identical, this means that our model does not have signs of overfitting or underfitting and generalizes well to our data. These results make sense given the nature of a naive bayes model. Since the model assumes conditional independance for all features given the class label, this simplifies the model and reduces the risk of overfitting as it doesn't learn any complex relationships. The probabalistic process of this model also makes it more robust to noise as it's learning the likelihood of the data instead of specific patterns. 

### Be sure to interpret your results! If you obtain a poor performance, compare it to some baseline easy task (i.e. random guessing, etc.) so you can have an estimate as to where your model performance is at. Propose various points of improvement for your model, and be thorough! Do not just say "we could use more data", or "I'll be sure to use reinforcement learning for the next milestone so we can better model our objective." Carefully work through your data preprocessing, your training steps, and point out any simplifications that may have impacted model performance, or perhaps any potential errors or biases in the dataset. You are not required to implement these points of improvement unless it is clear that your original model is significantly lacking in detail or effort.

Interpretation of Results:

Our Categorical Naive Bayes model achieved an accuracy of 0.8182, meaning it correctly classified approximately 82% of user recommendations based on the available features.

Points of Improvement:

- Feature selection: We could imrpove our feature selection process by using L1 regularization in order to pick the features that would help recommend games the best as we currently just chose the ones we believed would be the most impactful. This would force the coefficients of less important features to shrink to 0 and remove them from the model. Then, we would be left with the most significant features that would improve our model performance. 
- Feature engineering: We could create new interaction features to capture relationships between different features such as the ratio of playtime to price to see if the game is a good deal. 
- Data Preprocessing: Currently in the data preprocessing, we drop all rows that contain NaN values which works due to the large quantity of the data we have. However, in the future we could consider imputing missing values instead of dropping them, especially for important features. We could use the mode or other techniques like nearest neighbor imputation to fill missing values based on the closest games.
- Handling Outliers: Although Categorical Naive Bayes does not explicitly assume normal distribution like Gaussian Naive Bayes, extreme outliers in categorical features may still have impacted our model performance. We could look at the frequency distribution of categorical values in our EDA and identify rare categories that might skew results. If outliers are found, we could merge them into broader categories or apply techniques like smoothing to prevent overfitting to rare classes.


## Citations:

We used ChatGPT to help us understand Categorical Naive Bayes and how to implement it with our dataset.
