# Assignment 5, Part 2: Reinforcement Learning Theory

Setup for both questions: vocabulary {I, like, pizza}, episodes are exactly 3 words,
only "I like pizza" pays +10, all other sentences pay 0, gamma = 1.

## Question 1

### 1. Possible 1-word and 2-word states

After one word there are 3 states:

    [I], [like], [pizza]

After two words there are 3 x 3 = 9 states:

    [I, I]      [I, like]      [I, pizza]
    [like, I]   [like, like]   [like, pizza]
    [pizza, I]  [pizza, like]  [pizza, pizza]

Total non-terminal states reachable after at least one word: 3 + 9 = 12.

### 2. Terminal states

Each of the 3 positions is filled independently from 3 words, so there are

    3^3 = 27

terminal states. Exactly 1 of them ("I like pizza") gives a non-zero reward, so 26 give 0.

### 3. Value functions

The policy is uniform random, so each word is chosen with probability 1/3 at every step.
Reward arrives only at the end and gamma = 1, so the value of a state is just the
probability of finishing at "I like pizza" times 10:

    V(s) = P(reach "I like pizza" from s) * 10

(a) s0 = [ ]

All three words still have to come out right:

    P = (1/3) * (1/3) * (1/3) = 1/27
    V(s0) = (1/27) * 10 = 10/27 = 0.370

(b) s1 = [I]

The first word is already correct. Two words left, both must match:

    P = (1/3) * (1/3) = 1/9
    V(s1) = (1/9) * 10 = 10/9 = 1.111

(c) s2 = [I, like]

Only "pizza" is left to draw:

    P = 1/3
    V(s2) = (1/3) * 10 = 10/3 = 3.333

(d) s3 = [I, pizza]

The second word should have been "like", so the goal sentence can no longer be reached
no matter what comes next:

    P = 0
    V(s3) = 0

Summary:

    V([ ])         = 10/27 = 0.370
    V([I])         = 10/9  = 1.111
    V([I, like])   = 10/3  = 3.333
    V([I, pizza])  = 0

The values rise as the agent gets closer to the goal, which is what we expect.

## Question 2

### 1. Q-learning update rule

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The bracket is the temporal-difference error: the difference between the observed
estimate (r plus the discounted best value of the next state) and the current estimate.

### 2. Updated value of Q([I], like)

Given values:

    Q([I], like) = 1.0
    alpha = 0.5
    gamma = 1

Taking action "like" from state [I] moves the agent to s' = [I, like].
That state holds 2 words, and the episode is 3 words long, so it is not terminal.
Reward is only paid at the end of a sentence, so

    r = 0

The best next-state Q-value:

    max_a' Q([I, like], a') = max(1.0, 0.5, 2.0) = 2.0

Substituting:

    Q([I], like) <- 1.0 + 0.5 * [ 0 + 1 * 2.0 - 1.0 ]
                  = 1.0 + 0.5 * [ 1.0 ]
                  = 1.0 + 0.5
                  = 1.5

Answer: Q([I], like) = 1.5

The value went up because the next state [I, like] looks promising: its best action
("pizza") is worth 2.0, which is higher than the current estimate of 1.0.
