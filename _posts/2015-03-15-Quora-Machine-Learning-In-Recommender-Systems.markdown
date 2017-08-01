---
layout: post
title: How exactly is machine learning used in recommendation engines?
date: 2015-03-15
description: My quora response
---

Before diving into the detail, I would like to say that Collaborative Filtering algorithms are inherently based on similarity metrics, whether it's in user/item space (as in neighbourhood models) or latent space (as in factor models).
To answer the main question "where do ML techniques are used in recommendation engines", I would like to answer for various recommender system models with some examples

#### Content Based Recommendation (CBR): 

CBR provides personalized recommendation by matching user’s interests with description and attributes of items. For CBR, we can use standard ML techniques like Logistic Regression, SVM, Decision tree etc. based on user and item features for making predictions for eg: extent of like or dislike. Then, we can easily convert the result to ranked recommendation

#### Collaborative filtering (CF)

Neighborhood models** are heuristics based models which uses similarity metrics, for eg : pearson similarity, cosine similarity,  for finding similar users and items. It is based on, very reasonable, heuristic that a person will like the items that are similar to previously liked items. Rating prediction in item based neighborhood models is given by weighted average of ratings on similar items as shown below

$$
\hat{r}_{u,i} = b_{u,i} + \frac{\sum_{j \in N(i,k,u)}{s_{i,j} (r_{u,j} -   b_{u,j})}}{\sum_{j \in N(i,k,u)} s_{i,j} }
$$

where, $$N(i, k, u)$$ is a set of k items that are similar to i and rated by the user $$u$$; $$ s_{i,j} $$ is a similarity function (cosine or pearson correlation).
As there is no learning involved in above equation, any ML guy will say that this sucks (although it works pretty well in practice). So, in a quest of 1 million bucks (Netflix challenge), some smart people (Yehuda Koren et al.) thought about it and reformulated it as

$$
\hat{r}_{u,i} = b_{u,i} + \sum_{j \in N(i,k,u)}{\theta_{i,j}^{u} (r_{u,j} -   b_{u,j})}
$$

Now any ML guy will say "*Ohhh wait, it looks like linear regression  with *$$ \theta_{i,j}^{u} $$  as *parameters*". Now the ML guy is happy :).

So, Instead of using ad-hoc heuristic based $$s_{i,j}$$ to weight the ratings, now  the weights, $$ \theta_{i,j}^{u} $$, are learned. Note that, it  was crucial in winning Netflix prize. This is just an instance, out of many, of ML in recommendation.


**Matrix Factorization** learns user and item latent factors ($$U$$ an $$V$$) by minimizing reconstruction error on observed ratings. Formally, in an optimization framework it is given as

$$

\begin{aligned} {\text{min}}\sum_{u,i} (r_{ui} - U_{u}^{T} V_{i})^{2} + \lambda (\left \| U \right \|_{2} ^{2} + \left \| V \right \|_{2}^{2})  \end{aligned}

$$

First of all, when there is an optimization technique involved, it's definitely a ML thing.
Let's make this more clear by converting it to our own favorite Linear regression problem. ***So if you fix any one of the latent factor, say $$U$$, then it becomes linear regression on $$V$$.*** This way of optimization is well known in literature as ALS(alternating Least Squares). Again, ML guy who knows linear regression is very happy :).

Bayesian ML people, who  not only want point estimates but also uncertainty of the estimates, will reformulate the same problem into probabilistic setting and learn in their own bayesian way.  For detail refer to the <a href="http://www.gatsby.ucl.ac.uk/~amnih/papers/bpmf.pdf" target="_blank"> paper.

Similarly, neural network guys have used Restricted Boltzmann machine for rating prediction (this was also crucial in winning Netflix challenge). For detail refer to the <a href="http://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf" target="_blank"> paper </a>.

#### Machine learning and Cold Start

Cold start is a situation when a recommender system doesn’t have any historical information about user or item and is unable to make personalized recommendations. Cold start is the worst nightmare of any recommender system researcher. So one way to deal with cold start is eliciting new user’s preferences via initial interview. However, interview based elicitation is not useful as user often get bored when they are asked a series of questions. Now, ML guy can use his decision tree knowledge to learn a model that smartly chooses a minimum set of the question while learning user's preference.

Furthermore, there is a vast literature on Learning to rank for recommendation. Although, Learning to rank shares  DNA with Information retrieval, its more ML technique.

In a nutshell, Machine learning is very common in recommendation algorithms. Hence, the use of ML in recommendation solely depend upon your objective and reformulating the problem into your favorite ML algorithm (smart people, sometimes, come up with revolutionary new learning algorithms!!! ).

