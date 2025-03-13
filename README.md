# Update


## How did you choose the observations to be 4 different things?

The 4 observations we chose are hours played, genres, price, and past recommendations. We chose these as our observations because they represent key factors influencing a user's gaming preferences.

- Hours Played: This represents user engagement. If a user plays a game for a long time, they probably enjoyed it.
- Genres: Genre preference is one of the most important factors in gaming because users tend to favor certain types of games.
- Price: Not all users will spend a lot of money when choosing games so we categorized prices into 4 bins to account for the affordability of games. 
- Past Recommendations: If a user previously recommended a game, it is likely they enjoyed it so this is a useful factor for predicting future preferences.


## What kind of observation values could occur?

Observation values that can occur are:
- Hours Played: This is categorized into three bins: Low, Medium, High based on how much time a user has spent on a game.
- Genres: One-hot encoded categorical values indicating whether a game belongs to a popular genre.
- Price: Binned into four categories: (0 - 10), (10 - 30), (30 - 60), and (60+), encoded as [0, 1, 2, 3].
- Past Recommendations: Binary variable indicating whether a user has previously recommended a game.


## What kind of hidden state values do you have? 

The hidden states in our Hidden Markov Model represent different user preference states that are not directly observed but predicted from the data. 
These states include:

- Casual Gamer: Prefers low-cost, casual games and does not play frequently.
- Hardcore Gamer: Plays for long hours and prefers more complex or higher priced games.
- Genre-Specific Gamer: Focuses on a particular genre such as RPGs or shooters.
- New User: Recently started playing and has fewer game recommendations.


## How did you train your HMM?

We trained our Hidden Markov Model by analyzing the dataset and turning them into meaningful states. For our training we:

Preprocessed the Data:

We dropped unnecessary columns (e.g., support email, URLs), encoded categorical variables like genre and price, and binned numerical values like price and hours played into different levels.

Trained the HMM:

We used the processed features as input to an HMM. We then experimented with different numbers of hidden states (n_components = [2, 3, 4, 5]) and analyzed model performance.
The model was trained using the Expectation-Maximization (EM) algorithm to learn state transitions and probabilities from the data. This happens within the GaussianHMM library.

Evaluating Model Performance:

We first split user recommendations into training and testing sets. The model was tested by predicting whether it could correctly recommend games that a user had marked as recommended. We measured accuracy by checking if the top 5 predicted games matched actual user recommendations.


## Can you explain what a Gaussian HMM is?

A Gaussian Hidden Markov Model (HMM) is a type of HMM where the emission probabilities (observations) follow a Gaussian (Normal) distribution rather than a discrete distribution. This allows the model to handle quantitative continuous features like hours played and price which we used as features in our model.


## How do its training algorithms work?

The training of an HMM involves two steps using the Expectation-Maximization (EM) algorithm, specifically the Baum-Welch algorithm:

- Expectation Step (E-Step): Calculates the probability of different sequences of hidden states given the observed data.
- Maximization Step (M-Step): Updates the transition and emission probabilities to maximize the likelihood of the observed data.

The model iterates over these steps until convergence.


# CSE150A_Project: Game Recommendation System
### By: Derrick Nguyen, Amelia Lei, Jahnavi Naik
## Explain what your AI agent does in terms of PEAS. What is the "world" like?

  Our AI agent is a game recommendation agent, that aims to recommmend a game to the user based on their preferences and recommendations of other games they have played in the past.
  In terms of PEAS, we have:
  
  Performance Measure: the feedback that the user gives from the games that have been recommended to them, and how satisfied they are with it. 
  
  Environment: the user profiles and their preferences, along with all of the game data. 
  
  Actuators: the games that the agent recommends to the user based on the preferences. 
  
  Sensors: the information about the game and users that the agent uses to generate personalized recommendations. This includes the game catalog such as genre and ratings and the user preferences such as previous games they’ve enjoyed, favorite genres, or their budget.

## What kind of agent is it? Goal based? Utility based? etc. 

  The agent is a utility agent because it makes recommendations that optimize for user preferences, increases user engagement, and maximizes the likelihood of users interacting with the recommended games.

## Describe how your agent is set up and where it fits in probabilistic modeling

 To code this agent, we are using a hidden markov model. We chose the observations to be hours played, genres, price, and past recommendations (games that the user has already recommended). With these observations, the model predicts the current state, and recommends games to match it. The model fits into proabilisitic modeling because it represents the user's gaming preferences as the hidden states, and each state has an observable feature. It then uses the transition between states to update the recommendations to so they are personalized to the user.

## Train your first model

Our preprocessing, data exploration, and training can be found in this notebook: https://github.com/derricknguyen12/cse150a_project/blob/Milestone-2/proj.ipynb

For our preprocessing, we dropped rows that contained NaN values as those rows do not contribute significantly to our model and can be removed. We also dropped irrelevant columns that we would not be using in our model such as 'Support email' and 'Support Url'. Since 'Release date' was initally of type string, we converted it to datetime for easier manipulation. Before training our Hidden Markov Model, we label encoded the 'Genre' column. We also binned 'Price' into 4 bins: (0 - 10), (10 - 30), (30 - 60), and (60+) and labeled it with [0, 1, 2, 3], corresponding to how expensive the game is. Additionally, we binned 'hours' into 4 bins as well based on it's quantile: (min - Q1), (Q1 - Q2), (Q2 - Q3), and (Q3 - max) and labeled it with [0, 1, 2, 3], corresponding to different levels of hours played.

## Evaluate your model

  To evaluate our model, we trained it on user data by splitting the recommended games into training and testing sets. Our goal was to determine whether the model could successfully recommend games that a user had already marked as recommended. To measure accuracy, we analyzed the top five games suggested by the model for each user and checked whether the user had actually recommended any of them in the testing set. This allowed us to assess how well the model aligns with user preferences based on past recommendations. We obtained an accuracy score of 0.88. 
  
  To test whether our model was overfitting or underfitting, we created a fitting graph using models with 4 different n_components: [2, 3, 4, 5]. From our fitting graph, we found that our model was overfitting as the training accuracy was higher than the testing accuracy. 

## Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

  With an accuracy of 88.5%, we can conclude that our model is mostly predicting user predictions correctly when looking at hours of game play by the user, price, and game genre. Right now, we are using genre as a categorical variable without taking into account the user-specific aspects of genre as each user may have genres they prefer and play more frequently. To improve our model, we could use feature engineering to extract more information about the user to make predictions such as user-specific genre preferences that represent the genres each user has interacted with or played the most. We can also work to optimize our model, such as optimizing the number of hidden states that the model is using.

## Citations:

We used ChatGPT to help us understand Gaussian HMM and how to implement it with our dataset.
