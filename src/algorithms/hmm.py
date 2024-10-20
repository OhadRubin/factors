from ..core.factor_graph import FactorGraph,Variable, Factor


"""
2. EM Algorithm for HMMs (Baum-Welch)

The Expectation-Maximization (EM) algorithm, also known as the Baum-Welch algorithm
for Hidden Markov Models (HMMs), can be represented using factor graphs. This
approach provides a visual and intuitive understanding of the model's structure.

Factor Graph Structure:
1. Variables:
   - Hidden state variables at each time step.
   - Domain: Set of possible states in the HMM.

2. Factors:
   - Transition Factors: Connect consecutive hidden states, representing
     the probability of transitioning from one state to another.
   - Emission Factors: Connect hidden states to observed emissions, representing
     the probability of observing a particular output given a hidden state.

Implementation Outline:
The factor graph for an HMM can be constructed as follows:

**EM Algorithm Implementation**:
- Use the factor graph to perform **forward-backward** message passing.
- Update the transition and emission probabilities based on expected counts computed from the beliefs.
"""


def hmm_factor_graph(observations, states, transition_probs, emission_probs):
    fg = FactorGraph()
    variables = []
    # Create state variables for each time step
    for t in range(len(observations)):
        var = Variable(f"X_{t}", states)
        fg.add_variable(var)
        variables.append(var)
    # Create factors for transitions and emissions
    for t in range(len(observations)):
        state_var = variables[t]

        # Emission factor
        def emission_potential(assignment):
            state = assignment[state_var.name]
            observation = observations[t]
            return emission_probs[state][observation]

        emission_factor = Factor(f"E_{t}", [state_var], emission_potential)
        fg.add_factor(emission_factor)
        if t > 0:
            prev_state_var = variables[t - 1]

            # Transition factor
            def transition_potential(assignment):
                prev_state = assignment[prev_state_var.name]
                curr_state = assignment[state_var.name]
                return transition_probs[prev_state][curr_state]

            transition_factor = Factor(
                f"T_{t}", [prev_state_var, state_var], transition_potential
            )
            fg.add_factor(transition_factor)
    # Implement inference for the EM algorithm
    return fg
