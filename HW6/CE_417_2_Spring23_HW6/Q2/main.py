import numpy as np
from hmmlearn import hmm

states = ["Cold", "Medium", "Hot"]
n_states = len(states)

observations = ["Cooler Off", "Cooler On"]
n_observations = len(observations)

start_probability = np.array([float(num) for num in input().split()])

transition_probability = []
for i in range(n_states):
    transition_rows = [float(num) for num in input().split()]
    transition_probability.append(transition_rows)
transition_probability = np.array(transition_probability)

emission_probability = []
for i in range(n_states):
    emission_rows = [float(num) for num in input().split()]
    emission_probability.append(emission_rows)
emission_probability = np.array(emission_probability)

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
model2 = hmm.GaussianHMM(n_components=n_states)

observations_sequence = np.array([int(num) for num in input().split()]).reshape(-1, 1)

hidden_states = model.predict(observations_sequence)
most_likely_states = []
for i in range(len(hidden_states)):
    if hidden_states[i] == 0:
        most_likely_states.append("cold")
    elif hidden_states[i] == 1:
        most_likely_states.append("normal")
    elif hidden_states[i] == 2:
        most_likely_states.append("hot")
    else:
        pass

print('Most likely sequence of states:\n[', end='')
for i in range(len(most_likely_states)):
    if i == len(most_likely_states) - 1:
        print(most_likely_states[i], end=']\n')
    else:
        print(most_likely_states[i], end=' ')

after_observing_probs = model.predict_proba(observations_sequence)

print('Probability of states after observing:\n[', end='')
for i in range(len(after_observing_probs[-1])):
    if i == len(after_observing_probs[-1]) - 1:
        print(round(after_observing_probs[-1][i], 3), end=']\n')
    else:
        print(round(after_observing_probs[-1][i], 3), end=' ')
