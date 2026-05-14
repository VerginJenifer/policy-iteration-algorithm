# POLICY ITERATION ALGORITHM

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM

### Step 1:
Import libraries

### Step 2:
Create the FrozenLake environment, define actions (LEFT, DOWN, RIGHT, UP), and start with an initial policy that always moves RIGHT.

### Step 3:
Use policy_evaluation() to calculate the value of each state using the Bellman Expectation Equation until the values converge.

### Step 4:
Use policy_improvement() to compute Q-values for all actions in every state and choose the action with the highest value to form a better policy.

### Step 5:
In policy_iteration(), repeatedly evaluate the current policy and improve it until the policy no longer changes.

### Step 6:
When the policy becomes stable, return the optimal state-value function (optimal_V) and the optimal policy (optimal_pi) for the FrozenLake environment.

## POLICY IMPROVEMENT FUNCTION
### Name D Vergin Jenifer
### Register Number 212223240174
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[0])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a]+=prob*reward+gamma*V[next_state]*(not done)
    new_pi=lambda s:{s: a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi

```
## POLICY ITERATION FUNCTION
### Name D Vergin Jenifer
### Register Number 212223240174
```python

def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  pi=lambda s: {s: a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi= {s:pi(s) for s in range(len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi == {s: pi(s) for s in range(len(P))}:
      break
  return V, pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

<img width="747" height="282" alt="image" src="https://github.com/user-attachments/assets/a9f01cd2-ad2d-44d7-851d-81d797d0cb01" />
<img width="467" height="153" alt="image" src="https://github.com/user-attachments/assets/73762d1c-4868-4e26-bb7b-acaf9422ea59" />


### 2. Policy, Value function and success rate for the Improved Policy

<img width="462" height="167" alt="image" src="https://github.com/user-attachments/assets/e7bd1633-d047-4a57-b3f0-de12c8e6e67a" />
<img width="662" height="40" alt="image" src="https://github.com/user-attachments/assets/130ebdb3-410c-471f-a35b-4f94eae89050" />



### 3. Policy, Value function and success rate after policy iteration

<img width="482" height="151" alt="image" src="https://github.com/user-attachments/assets/057d62a4-e67e-4aa2-a003-e7f44cd6e148" />




## RESULT:

The Policy Iteration algorithm successfully converged to an optimal policy for the Frozen Lake environment. The optimal policy achieved a higher success rate and improved average return compared to the initial policy.
