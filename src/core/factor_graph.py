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
        var_names = ", ".join([var.name for var in self.variables])
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
        return (
            f"FactorGraph(variables={len(self.variables)}, factors={len(self.factors)})"
        )


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
                    incoming = [
                        self.messages[(v, factor)] for v in factor.variables if v != var
                    ]
                    message = self.compute_factor_to_var_message(factor, var, incoming)
                    self.messages[(factor, var)] = message
            # Update messages from variables to factors
            for var in self.graph.variables:
                for factor in var.neighbors:
                    # Compute the message from variable to factor
                    incoming = [
                        self.messages[(f, var)] for f in var.neighbors if f != factor
                    ]
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
                assignment = {
                    v.name: d[idx]
                    for v, d, idx in zip(other_vars, domains, assignments)
                }
                assignment[var.name] = val
                # Compute factor potential
                potential = factor.potential_func(assignment)
                # Multiply by incoming messages
                prod_incoming = np.prod(
                    [msg[idx] for msg, idx in zip(incoming_messages, assignments)]
                )
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
