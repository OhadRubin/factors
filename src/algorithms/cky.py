from ..core.factor_graph import FactorGraph, Variable, Factor


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
        for j in range(i + 1, n + 1):
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
                            if (
                                assignment[var_i_j.name] == rule.lhs
                                and assignment[var_i_k.name] == rule.rhs1
                                and assignment[var_k_j.name] == rule.rhs2
                            ):
                                return 1
                            else:
                                return 0

                        factor = Factor(
                            f"F_{i}_{k}_{j}",
                            [var_i_j, var_i_k, var_k_j],
                            potential_func,
                        )
                        fg.add_factor(factor)
    # Implement inference to find the best parse
    return fg
