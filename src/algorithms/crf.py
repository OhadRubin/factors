from src.core.factor_graph import FactorGraph, Variable, Factor
import numpy as np


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
                return np.exp(weights["trans"][(prev_label, curr_label)])

            transition_factor = Factor(
                f"T_{t}", [prev_label_var, label_var], transition_potential
            )
            fg.add_factor(transition_factor)
    return fg
