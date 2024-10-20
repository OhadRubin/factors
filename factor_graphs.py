"""
Factor Graph Abstraction for Natural Language Processing

This module provides a Python implementation of factor graphs, a powerful framework
for representing and solving probabilistic inference problems. Factor graphs are
particularly useful in NLP for tasks such as:

- CKY parsing
- Expectation Maximization (EM) for Hidden Markov Models (HMMs)
- Conditional Random Fields (CRFs)

The abstraction consists of three main components:

1. Variable Node: Represents a variable in the factor graph.
2. Factor Node: Represents a factor (function) over one or more variables.
3. Factor Graph: Represents the entire graph, consisting of variable and factor nodes
   connected by edges.

This implementation allows for a unified approach to various NLP algorithms,
providing a flexible and extensible framework for probabilistic inference tasks.
# ## Advantages of This Abstraction

- **Modularity**: The abstraction separates variables, factors, and inference algorithms, making it easy to modify or extend components.
- **Reusability**: Common components can be reused across different algorithms.
- **Flexibility**: Supports both directed and undirected graphical models.
- **Extendibility**: New types of variables, factors, and inference methods can be added.


"""

# Import necessary libraries
import numpy as np
from collections import defaultdict

class Variable:
    def __init__(self, name, domain):
        """
        Initialize a variable node.
        :param name: Name of the variable.
        :param domain: Possible values the variable can take.
        """
        self.name = name
        self.domain = domain
        self.neighbors = []  # Connected factors

    def __repr__(self):
        return f"Variable({self.name})"

class Factor:
    def __init__(self, name, variables, potential_func):
        """
        Initialize a factor node.
        :param name: Name of the factor.
        :param variables: List of variables the factor is connected to.
        :param potential_func: Function defining the factor's potential.
        """
        self.name = name
        self.variables = variables  # Variables connected to this factor
        self.potential_func = potential_func  # Potential function
        for var in variables:
            var.neighbors.append(self)

    def __repr__(self):
        var_names = ', '.join([var.name for var in self.variables])
        return f"Factor({self.name}: {var_names})"

class FactorGraph:
    def __init__(self):
        """
        Initialize the factor graph.
        """
        self.variables = []
        self.factors = []

    def add_variable(self, variable):
        self.variables.append(variable)
    
    def add_factor(self, factor):
        self.factors.append(factor)

    def __repr__(self):
        return f"FactorGraph(variables={len(self.variables)}, factors={len(self.factors)})"

class FactorGraphInference:
    # ### Message Passing Mechanism
    # For inference, we'll implement a simple sum-product message passing algorithm (belief propagation).
    def __init__(self, factor_graph):
        self.graph = factor_graph
        self.messages = {}  # Messages between variables and factors

    def initialize_messages(self):
        # Initialize messages to uniform distributions
        self.messages = {}
        for factor in self.graph.factors:
            for var in factor.variables:
                self.messages[(factor, var)] = np.ones(len(var.domain))
                self.messages[(var, factor)] = np.ones(len(var.domain))
    
    def run_belief_propagation(self, max_iters=10):
        self.initialize_messages()
        for iteration in range(max_iters):
            # Update messages from factors to variables
            for factor in self.graph.factors:
                for var in factor.variables:
                    # Compute the message from factor to variable
                    incoming = [self.messages[(v, factor)] for v in factor.variables if v != var]
                    message = self.compute_factor_to_var_message(factor, var, incoming)
                    self.messages[(factor, var)] = message
            # Update messages from variables to factors
            for var in self.graph.variables:
                for factor in var.neighbors:
                    # Compute the message from variable to factor
                    incoming = [self.messages[(f, var)] for f in var.neighbors if f != factor]
                    message = self.compute_var_to_factor_message(var, incoming)
                    self.messages[(var, factor)] = message

    def compute_factor_to_var_message(self, factor, var, incoming_messages):
        # Compute the message from a factor to a variable
        other_vars = [v for v in factor.variables if v != var]
        domains = [v.domain for v in other_vars]
        idx_var = factor.variables.index(var)
        var_domain = var.domain

        # Initialize outgoing message
        message = np.zeros(len(var_domain))
        # Iterate over possible values of the variable
        for i, val in enumerate(var_domain):
            total = 0
            # Sum over all combinations of other variables
            for assignments in np.ndindex(*[len(d) for d in domains]):
                assignment = {v.name: d[idx] for v, d, idx in zip(other_vars, domains, assignments)}
                assignment[var.name] = val
                # Compute factor potential
                potential = factor.potential_func(assignment)
                # Multiply by incoming messages
                prod_incoming = np.prod([msg[idx] for msg, idx in zip(incoming_messages, assignments)])
                total += potential * prod_incoming
            message[i] = total
        # Normalize message
        message /= np.sum(message)
        return message

    def compute_var_to_factor_message(self, var, incoming_messages):
        # Multiply incoming messages and normalize
        message = np.prod(incoming_messages, axis=0)
        message /= np.sum(message)
        return message

    def compute_beliefs(self):
        # Compute marginal beliefs for variables
        beliefs = {}
        for var in self.graph.variables:
            incoming = [self.messages[(factor, var)] for factor in var.neighbors]
            belief = np.prod(incoming, axis=0)
            belief /= np.sum(belief)
            beliefs[var.name] = belief
        return beliefs

"""
Implementing Algorithms Using the Factor Graph Abstraction

1. CKY Parsing as a Factor Graph

The CKY (Cocke-Younger-Kasami) parsing algorithm can be elegantly represented
using our factor graph framework. In this representation:

- Variables correspond to possible non-terminals at each span in the parse chart.
- Factors represent the grammar rules that connect these variables.

Key Components:

1. Variables: 
   - One for each possible span in the input sentence.
   - Domain: Set of non-terminal symbols that can be generated at that span.

2. Factors: 
   - Based on Context-Free Grammar (CFG) production rules.
   - Connect variables of smaller spans to those of larger spans.

Implementation Outline:
"""
def cky_factor_graph(words, grammar_rules):
    fg = FactorGraph()
    n = len(words)
    # Create variables for each position and span
    variables = {}
    for i in range(n):
        for j in range(i+1, n+1):
            var_name = f"X_{i}_{j}"
            # Domain is the set of possible non-terminals
            domain = get_possible_non_terminals(grammar_rules)
            var = Variable(var_name, domain)
            fg.add_variable(var)
            variables[(i, j)] = var
    # Create factors based on grammar rules
    for (i, k), var_i_k in variables.items():
        for (k, j), var_k_j in variables.items():
            if k == j:
                continue
            for (i, j), var_i_j in variables.items():
                if i == k or j <= k:
                    continue
                # Check if any production rule applies
                for rule in grammar_rules:
                    # if rule can produce var_i_j from var_i_k and var_k_j:
                    if rule.can_produce(var_i_j, var_i_k, var_k_j):
                        def potential_func(assignment):
                            # Return 1 if the rule applies, 0 otherwise
                            # return 1 if rule applies to assignment else 0
                            if assignment[var_i_j.name] == rule.lhs and assignment[var_i_k.name] == rule.rhs1 and assignment[var_k_j.name] == rule.rhs2:
                                return 1
                            else:
                                return 0
                        factor = Factor(f"F_{i}_{k}_{j}", [var_i_j, var_i_k, var_k_j], potential_func)
                        fg.add_factor(factor)
    # Implement inference to find the best parse
    return fg


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
            transition_factor = Factor(f"T_{t}", [prev_state_var, state_var], transition_potential)
            fg.add_factor(transition_factor)
    # Implement inference for the EM algorithm
    return fg


# ### 3. Conditional Random Fields (CRFs)


def crf_factor_graph(features, labels, weights, compute_score):
    """
    Construct a factor graph for a Conditional Random Field (CRF).

    CRFs extend HMMs by considering observations as part of the graph and defining
    factors over both states and observations.

    Variables:
    - Label variables for each position.

    Domain:
    - Set of possible labels.

    Factors:
    - Transition Factors: Encode the dependency between consecutive labels.
    - Feature Factors: Incorporate features from the observations.

    Implementation Outline:
    The factor graph for a CRF is constructed as follows:
    1. Create label variables for each position in the sequence.
    2. Add feature factors to incorporate observations and their weights.
    3. Add transition factors to model dependencies between consecutive labels.

    Args:
        features (list): List of feature vectors for each position in the sequence.
        labels (list): List of possible labels (the domain of each variable).
        weights (dict): Dictionary of weights for features and transitions.
        compute_score: Function to compute the score of a feature vector and label.

    Returns:
        FactorGraph: A factor graph representing the CRF structure.

    Note:
    After constructing the factor graph, use belief propagation for inference and learning.
    """
    fg = FactorGraph()
    variables = []
    # Create label variables for each position
    for t in range(len(features)):
        var = Variable(f"Y_{t}", labels)
        fg.add_variable(var)
        variables.append(var)
    # Create factors
    for t in range(len(features)):
        label_var = variables[t]
        # Feature factor
        def feature_potential(assignment):
            label = assignment[label_var.name]
            feature_vector = features[t]
            score = compute_score(feature_vector, label, weights)
            return np.exp(score)
        feature_factor = Factor(f"F_{t}", [label_var], feature_potential)
        fg.add_factor(feature_factor)
        if t > 0:
            prev_label_var = variables[t - 1]
            # Transition factor
            def transition_potential(assignment):
                prev_label = assignment[prev_label_var.name]
                curr_label = assignment[label_var.name]
                return np.exp(weights['trans'][(prev_label, curr_label)])
            transition_factor = Factor(f"T_{t}", [prev_label_var, label_var], transition_potential)
            fg.add_factor(transition_factor)
    return fg


# Define HMM parameters
states = ['Rainy', 'Sunny']
observations = ['walk', 'shop', 'clean']
start_probs = {'Rainy': 0.6, 'Sunny': 0.4}
transition_probs = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6}
}
emission_probs = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1}
}

# Observed sequence
obs_sequence = ['walk', 'shop', 'clean']

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