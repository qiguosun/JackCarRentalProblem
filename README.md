# Jack's Car Rental Problem: A example solved by Policy Iteration

The problem is given in the book by [Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/ebook/the-book.html), the aims is to show the policy iteration algorithm in reinforcement learning.

## Problem statement

Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. 
If Jack has a car available, he rents it out and is credited \$ 10 by the national company. If he is out of cars at that location, then the business is lost. 
Cars become available for renting the day after they are returned. To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of \$ 2 per car moved. 

We assume that the number of cars requested and returned at each location are Poisson random variables, meaning that the probability that the number is n is $P(X=n)=\frac{\lambda ^n}{n!} e^{- \lambda}$, where $\lambda$ is the expected number. Suppose $\lambda$ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be $\gamma=0.9$ and formulate this as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight. 

## Run the code

First you could use virtual environment to install the required libraries:
```python
pip install -r dependencies.txt
```

To run the demo, you can run the results.py

```python
python results.py
```

## Implementation details

1) The file poissonDistribution.py is used to generate poisson Distribution of cars rent and return.

2) For environment.py file, the init() method initialized poisson distributions; the calc_reward_and_prob method calculate the reward and probility of each state.

3) The policyIteration.py follows the pseudocode of policy iteration described in the book, which is used to find the optimal policy.

4) The results.py file is used to illustrate the optimal policy.

## License

[MIT](https://choosealicense.com/licenses/mit/)


