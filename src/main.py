from src.algorithms.hmm import hmm_factor_graph
from src.core.factor_graph import FactorGraphInference


# Define HMM parameters
states = ["Rainy", "Sunny"]
observations = ["walk", "shop", "clean"]
start_probs = {"Rainy": 0.6, "Sunny": 0.4}
transition_probs = {
    "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
    "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
}
emission_probs = {
    "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
    "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
}

# Observed sequence
obs_sequence = ["walk", "shop", "clean"]

# Create the factor graph
fg = hmm_factor_graph(obs_sequence, states, transition_probs, emission_probs)

# Perform inference
inference = FactorGraphInference(fg)
inference.run_belief_propagation(max_iters=5)
beliefs = inference.compute_beliefs()

# Display the results
for var_name, belief in beliefs.items():
    print(f"Belief for {var_name}:")
    for state, prob in zip(states, belief):
        print(f"  P({state}) = {prob:.4f}")
