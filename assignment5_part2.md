# Assignment 5, Part 2: Reinforcement Learning Theory

Setup for both questions. The vocabulary is I, like, pizza. Each episode is 3 words.
Only "I like pizza" gives a reward of +10. All other sentences give 0. Gamma = 1.

## Question 1

### 1. Possible 1-word and 2-word states

There are 3 one-word states:

    [I], [like], [pizza]

There are 3 x 3 = 9 two-word states:

    [I, I], [I, like], [I, pizza],
    [like, I], [like, like], [like, pizza],
    [pizza, I], [pizza, like], [pizza, pizza]

### 2. Terminal states

Each of the 3 positions has 3 choices, so there are

    3^3 = 27

terminal states. Only 1 of them gives a reward. The other 26 give 0.

### 3. Value functions

Each word is picked with probability 1/3. The reward comes only at the end, and gamma = 1.
So the value of a state is the chance of ending at "I like pizza" times 10.

    V(s) = P(reach "I like pizza" from s) x 10

(a) s0 = [ ]

All 3 words must be right.

    P = (1/3)(1/3)(1/3) = 1/27
    V(s0) = 10/27 = 0.370

(b) s1 = [I]

The first word is right. Two words are left.

    P = (1/3)(1/3) = 1/9
    V(s1) = 10/9 = 1.111

(c) s2 = [I, like]

Only pizza is left.

    P = 1/3
    V(s2) = 10/3 = 3.333

(d) s3 = [I, pizza]

The second word is wrong, so the goal can no longer be reached.

    P = 0
    V(s3) = 0

Summary:

    V([ ]) = 10/27 = 0.370
    V([I]) = 10/9 = 1.111
    V([I, like]) = 10/3 = 3.333
    V([I, pizza]) = 0

The value rises as the agent gets closer to the goal.

## Question 2

### 1. Q-learning update rule

    Q(s, a) = Q(s, a) + alpha [ r + gamma max Q(s', a') - Q(s, a) ]

The term in brackets is the temporal difference error.

### 2. Updated value of Q([I], like)

Given: Q([I], like) = 1.0, alpha = 0.5, gamma = 1.

Taking "like" from [I] leads to s' = [I, like]. That state has 2 words, so the sentence
is not finished. The reward is 0.

    max Q([I, like], a') = max(1.0, 0.5, 2.0) = 2.0

Substituting:

    Q([I], like) = 1.0 + 0.5 [ 0 + 2.0 - 1.0 ]
                 = 1.0 + 0.5 (1.0)
                 = 1.5

Answer: Q([I], like) = 1.5
