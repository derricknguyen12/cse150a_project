# CSE150A_Project: Game Recommendation System
### By: Derrick Nguyen, Amelia Lei, Jahnavi Naik

## PEAS/Agent Analysis:

### Describe your agent in terms of PEAS and give a background of your task at hand.

Our AI agent is a game recommendation agent, that aims to recommmend a game to the user based on their preferences and recommendations of other games they have played in the past.
  In terms of PEAS, we have:
  
  Performance Measure: the feedback that the user gives from the games that have been recommended to them, and how satisfied they are with it. 
  
  Environment: the user profiles and their preferences, along with all of the game data. 
  
  Actuators: the games that the agent recommends to the user based on the preferences. 
  
  Sensors: the information about the game and users that the agent uses to generate personalized recommendations. This includes the game catalog such as genre and ratings and the user preferences such as previous games they’ve enjoyed, favorite genres, or their budget.


## Agent Setup, Data Preprocessing, Training setup

### Give an exploration of your dataset, and highlight which variables are important. Give a brief overview of each variable and its role in your agent/model. (Draw a picture!!)

For our project, we used 2 datasets. One includes many steam games along with different attributes that describe them such as price, genre, release date, required age, about the game, etc. We also have a dataframe that has user informations, which tells us games that different users have played, if they recommend them, and and their reviews. After pre-processing these dataframes, we found that the important variables we would use are review ratio, is free, category count, genres, price tier, playtime tier, and community size. These are variables that we engineered using the avaibale features. Review ratio is a ratio of positive over negative reviews to determine the overall sentiment of the game. Is free tells us if the game is free to play or not. Category count tells us how many genres there are, which is useful as games with more genres tend to be more complex, giving us insight to what type of complexity of games users may prefer. Price tier places the games into categories of prices, playtime tier gives us categories of playtimes, and community size gives us categories of how large the community is, telling us how popular or underground the game is.

### Describe in detail how your variables interact with each other, and if your model fits a particular structure, explain why you chose that structure to model your agent. If it does not, further elaborate on why you chose that model based on the variables.

### Describe your process for calculating parameters in your model. That is, if you wish to find the CPTs, provide formulas as to how you computed them. If you used algorithms in class, just mention them.

### Important: if you used a library like pgmpy, etc., be sure to explain what it does. Then, cite your source (library homepage link or documentation is sufficient) at the bottom of your README.md file.
If you use a particular algorithm/model/structure not covered in this class as your core structure (i.e. GaussianHMM, etc.) you must carefully explain the formulation behind this structure. How does it differ from its discrete analog (if it has one?) How do you perform inference on it?

### If you use a particular algorithm/model/structure not covered in this class as a non-core structure (i.e. RandomHillCimb to create BN dependencies) you may briefly explain what the function does. Remember to provide a source.

### Be sure to link to a clean, documented portion of code in your notebook or provide a code snippet in the README.

## Conclusion/Results (20pts)

### Describe in detail your results, including any helpful visualizations like heatmaps, confusion matrices, etc. (if applicable). Please provide numerical results (unless your model's performance metrics do not include numbers).
### Be sure to interpret your results! If you obtain a poor performance, compare it to some baseline easy task (i.e. random guessing, etc.) so you can have an estimate as to where your model performance is at.
Propose various points of improvement for your model, and be thorough! Do not just say "we could use more data", or "I'll be sure to use reinforcement learning for the next milestone so we can better model our objective." Carefully work through your data preprocessing, your training steps, and point out any simplifications that may have impacted model performance, or perhaps any potential errors or biases in the dataset. You are not required to implement these points of improvement unless it is clear that your original model is significantly lacking in detail or effort.







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

  
  To test whether our model was overfitting or underfitting, we created a fitting graph using models with 4 different n_components: [2, 3, 4, 5]. From our fitting graph, we found that our model was overfitting as the training accuracy was higher than the testing accuracy. 

## Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?


## Citations:

We used ChatGPT to help us understand Gaussian HMM and how to implement it with our dataset.
