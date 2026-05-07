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

<img width="807" height="295" alt="image" src="https://github.com/user-attachments/assets/7fcecd7c-113a-45c0-be10-97fea952bd2c" />
<img width="498" height="150" alt="image" src="https://github.com/user-attachments/assets/c0c19c77-f695-4e2e-93f5-5e98ec5ff1a1" />


### 3. Policy, Value function and success rate after policy iteration

<img width="770" height="308" alt="image" src="https://github.com/user-attachments/assets/f27a1f25-7a1e-4b82-b7c3-45360ac593b9" />
<img width="752" height="143" alt="image" src="https://github.com/user-attachments/assets/3c1ea675-5691-4553-9103-a9be47e2ddbf" />



## RESULT:

The Policy Iteration algorithm successfully converged to an optimal policy for the Frozen Lake environment. The optimal policy achieved a higher success rate and improved average return compared to the initial policy.
