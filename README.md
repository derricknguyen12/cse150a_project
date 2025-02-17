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

 To code this agent, we are using a hidden markov model. We chose the observations to be hours played (categorized: low, medium, high), genres (one-hot encoded to determine if it is one of the top genres), price, and past recommendations(games that the user has already recommended). With these observations, the model predicts the current state, and recommends games to match it. The model fits into proabilisitic modeling because it represents the user's gaming preferences as the hidden states, and each state has an observable feature. It then uses the transition between states to update the recommendations to so they are personalized to the user.

## Train your first model

Our preprocessing, data exploration, and training can be found in this notebook: [View the Notebook](proj.ipynb)

For our preprocessing, we dropped rows that contained NaN values as those rows do not contribute significantly to our model and can be removed. We also dropped irrelevant columns that we would not be using in our model such as 'Support email' and 'Support Url'. Since 'Release date' was initally of type string, we converted it to datetime for easier manipulation. Before training our Hidden Markov Model, we label encoded the 'Genre' column and binned 'Price' into 4 bins: (0 - 10), (10 - 30), (30 - 60), and (60+) and labeled it with [0, 1, 2, 3], corresponding to how expensive the game is. 
  

## Evaluate your model

  To evaluate our model, we trained it on user data by splitting the recommended games into training and testing sets. Our goal was to determine whether the model could successfully recommend games that a user had already marked as recommended. To measure accuracy, we analyzed the top five games suggested by the model for each user and checked whether the user had actually recommended any of them in the testing set. This allowed us to assess how well the model aligns with user preferences based on past recommendations. We obtained an accuracy score of 0.87. 
  
  To test whether our model was overfitting or underfitting, we created a fitting graph using models with 4 different n_components: [2, 3, 4, 5]. From our fitting graph, we found that our model was overfitting as the training accuracy was higher than the testing accuracy. 

## Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

  With an accuracy of 87%, we can conclude that our model is mostly predicting user predictions correctly when looking at hours of game play by the user, price, and game genre. Right now, we are using genre as a categorical variable without taking into account the user-specific aspects of genre as each user may have genres they prefer and play more frequently. To improve our model, we could use feature engineering to extract more information about the user to make predictions such as user-specific genre preferences that represent the genres each user has interacted with or played the most. We can also work to optimize our model, such as optimizing the number of hidden states that the model is using.

