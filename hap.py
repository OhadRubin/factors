"""
Hierarchical Affinity Propagation (HAP)
---

### **Variables**

1. **Data Points and Layers**

   - **\( N \)**: Total number of data points.
   - **\( L \)**: Total number of layers in the hierarchy.
   - **Indices**:
     - **\( i, j, k \in \{1, 2, \dots, N\} \)**: Indices for data points.
     - **\( l \in \{1, 2, \dots, L\} \)**: Indices for layers.

2. **Hidden Variables**

   - **Assignment Variables**: \( h_{ij}^l \)
     - \( h_{ij}^l = 1 \) if data point \( i \) selects data point \( j \) as its exemplar at layer \( l \).
     - \( h_{ij}^l = 0 \) otherwise.
   - **Exemplar Indicators**: \( e_j^l \)
     - \( e_j^l = 1 \) if data point \( j \) is chosen as an exemplar at layer \( l \).
     - \( e_j^l = 0 \) otherwise.

---

### **Input Data**

1. **Pairwise Similarities**

   - **\( s_{ij}^l \)**: Similarity between data point \( i \) and data point \( j \) at layer \( l \).
     - Defined for all \( i, j \in \{1, 2, \dots, N\} \) and \( l \in \{1, 2, \dots, L\} \).
     - For all \( l \) and \( j \), \( s_{jj}^l = 0 \).

2. **Exemplar Preferences**

   - **\( c_j^l \)**: Preference for data point \( j \) to be chosen as an exemplar at layer \( l \).
     - Defined for all \( j \in \{1, 2, \dots, N\} \) and \( l \in \{1, 2, \dots, L\} \).

---

### **Functions in the Factor Graph**

1. **Similarity Function \( S_{ij}^l \)**

   - For all \( i, j, l \):
     \[
     S_{ij}^l(h_{ij}^l) = s_{ij}^l \cdot h_{ij}^l.
     \]

2. **Exemplar Preference Function \( C_j^l \)**

   - For all \( j, l \):
     \[
     C_j^l(e_j^l) = c_j^l \cdot e_j^l.
     \]

3. **Assignment Constraint Function \( I_i^l \)**

   - **For layer \( l = 1 \)** (the bottom layer), for all \( i \):
     \[
     I_i^1(h_{i1}^1, h_{i2}^1, \dots, h_{iN}^1) =
     \begin{cases}
     0, & \text{if } \sum_{j=1}^N h_{ij}^1 = 1, \\
     -\infty, & \text{otherwise}.
     \end{cases}
     \]
     - This ensures each data point assigns itself to exactly one exemplar at layer 1.

   - **For layers \( l = 2 \) to \( L \)**, for all \( i \):
     \[
     I_i^l(h_{i1}^l, h_{i2}^l, \dots, h_{iN}^l, e_i^{l-1}) =
     \begin{cases}
     0, & \text{if } \sum_{j=1}^N h_{ij}^l = e_i^{l-1}, \\
     -\infty, & \text{otherwise}.
     \end{cases}
     \]
     - This ensures that a data point at layer \( l \) assigns itself to an exemplar only if it was an exemplar at layer \( l - 1 \).

4. **Exemplar Consistency Function \( E_j^l \)**

   - For all \( j, l \):
     \[
     E_j^l(h_{1j}^l, h_{2j}^l, \dots, h_{Nj}^l, e_j^l) =
     \begin{cases}
     0, & \text{if } e_j^l = h_{jj}^l = \max_{i=1}^N h_{ij}^l, \\
     -\infty, & \text{otherwise}.
     \end{cases}
     \]
     - This ensures that \( j \) can be an exemplar at layer \( l \) only if it selects itself, and if any data point selects \( j \) as its exemplar, then \( j \) must be an exemplar.

---

### **Objective Function**

We aim to maximize:

\[
\begin{aligned}
\text{Objective} = & \sum_{l=1}^L \left[ \sum_{i=1}^N \sum_{j=1}^N S_{ij}^l(h_{ij}^l) + \sum_{j=1}^N C_j^l(e_j^l) \right] \\
& + \sum_{i=1}^N I_i^1(h_{i1}^1, h_{i2}^1, \dots, h_{iN}^1) \\
& + \sum_{l=2}^L \sum_{i=1}^N I_i^l(h_{i1}^l, h_{i2}^l, \dots, h_{iN}^l, e_i^{l-1}) \\
& + \sum_{l=1}^L \sum_{j=1}^N E_j^l(h_{1j}^l, h_{2j}^l, \dots, h_{Nj}^l, e_j^l).
\end{aligned}
\]

This objective function includes:

- **Similarity terms**: Rewards for assigning data points to similar exemplars.
- **Exemplar preference terms**: Rewards (or penalties) for choosing data points as exemplars.
- **Assignment constraints**: Ensures valid assignments at each layer.
- **Exemplar consistency constraints**: Maintains consistency of exemplar status.

---

### **Message-Passing Equations**

We use the max-sum algorithm to perform approximate MAP inference on the factor graph. Messages are passed between variable nodes and factor nodes.

#### **Initialization**

- Set all availability messages \( \alpha_{ij}^l = 0 \) for all \( i, j, l \).
- Set all downward messages \( \phi_j^0 = 0 \) for all \( j \).

#### **Iterative Updates**

For each iteration, perform the following steps until convergence:

**For each layer \( l = 1 \) to \( L \)**:

1. **Compute Adjusted Preferences \( \hat{c}_j^l \)**

   - For all \( j \):
     \[
     \hat{c}_j^l = c_j^l + \phi_j^{l-1}.
     \]
     - **Note**: For \( l = 1 \), \( \phi_j^0 = 0 \).

2. **Compute Responsibility Messages \( \rho_{ij}^l \)**

   - **For layer \( l = 1 \)**, for all \( i, j \):
     \[
     \rho_{ij}^1 = s_{ij}^1 - \max_{k \ne j} \left( \alpha_{ik}^1 + s_{ik}^1 \right).
     \]
   - **For layers \( l = 2 \) to \( L \)**, for all \( i, j \):
     \[
     \rho_{ij}^l = s_{ij}^l + \min \left[ \tau_i^l, -\max_{k \ne j} \left( \alpha_{ik}^l + s_{ik}^l \right) \right],
     \]
     where \( \tau_i^l \) is defined below.

3. **Compute Availability Messages \( \alpha_{ij}^l \)**

   - For all \( i, j \):
     \[
     \alpha_{ij}^l =
     \begin{cases}
     \hat{c}_j^l + \sum_{k \ne j} \max \left( 0, \rho_{kj}^l \right), & \text{if } i = j, \\
     \min \left[ 0, \hat{c}_j^l + \rho_{jj}^l + \sum_{k \notin \{i, j\}} \max \left( 0, \rho_{kj}^l \right) \right], & \text{if } i \ne j.
     \end{cases}
     \]

4. **Compute Upward Messages \( \tau_j^{l+1} \)**

   - **For layers \( l = 1 \) to \( L - 1 \)**, for all \( j \):
     \[
     \tau_j^{l+1} = \hat{c}_j^l + \rho_{jj}^l + \sum_{k \ne j} \max \left( 0, \rho_{kj}^l \right).
     \]
     - This message is sent from \( e_j^l \) to \( I_j^{l+1} \) in layer \( l + 1 \).

5. **Compute Downward Messages \( \phi_j^l \)**

   - **For layers \( l = L \) down to \( l = 1 \)**, for all \( j \):
     \[
     \phi_j^l = \max_{k=1}^N \left( \alpha_{jk}^l + s_{jk}^l \right).
     \]
     - For \( l = 1 \), \( \phi_j^1 \) is computed but not used in \( \hat{c}_j^1 \).

6. **Update Adjusted Preferences for Next Iteration**

   - For all \( j \) and \( l \):
     \[
     \hat{c}_j^l = c_j^l + \phi_j^{l-1}.
     \]

#### **Convergence Criteria**

- Repeat steps 1-6 until the messages stabilize (changes are below a threshold) or a maximum number of iterations is reached.

---

### **Assignment Extraction**

After convergence:

1. **Determine Exemplars \( e_j^l \)**

   - For all \( j, l \):
     \[
     e_j^l =
     \begin{cases}
     1, & \text{if } \tau_j^{l+1} + \phi_j^l > 0, \\
     0, & \text{otherwise}.
     \end{cases}
     \]
     - For \( l = L \), \( \tau_j^{L+1} \) is not defined; we can set \( \tau_j^{L+1} = 0 \) or handle accordingly.

2. **Assign Data Points to Exemplars \( h_{ij}^l \)**

   - For all \( i, l \) where \( e_i^{l-1} = 1 \) (i.e., \( i \) is an exemplar at layer \( l - 1 \)):
     - Assign \( i \) to the exemplar \( j \) that maximizes:
       \[
       h_{ij}^l =
       \begin{cases}
       1, & \text{if } j = \arg\max_{k} \left( \alpha_{ik}^l + \rho_{ik}^l + s_{ik}^l \right), \\
       0, & \text{otherwise}.
       \end{cases}
       \]
   - For \( i \) where \( e_i^{l-1} = 0 \), set \( h_{ij}^l = 0 \) for all \( j \).

---

### **Efficient Computation**

- **Time Complexity**: \( \mathcal{O}(L N^2) \) per iteration.
- **Optimization Tips**:
  - Precompute \( \max_{k \ne j} (\alpha_{ik}^l + s_{ik}^l) \) for all \( i \) when computing \( \rho_{ij}^l \).
  - Precompute \( \sum_{k \ne j} \max(0, \rho_{kj}^l) \) for all \( j \) when computing \( \alpha_{ij}^l \).
  - Use sparse data structures if \( s_{ij}^l \) is sparse.

---

### **Implementation Notes**

- **Data Structures**:
  - Use multidimensional arrays or tensors to store messages \( \alpha_{ij}^l \) and \( \rho_{ij}^l \).
  - Indexing should be carefully handled to avoid off-by-one errors.

- **Numerical Stability**:
  - **Damping**: To prevent oscillations, update messages using damping:
    \[
    \text{new message} = (1 - \lambda) \times \text{old message} + \lambda \times \text{computed message},
    \]
    where \( \lambda \in [0, 1] \) is the damping factor (e.g., \( \lambda = 0.5 \)).

- **Convergence Check**:
  - Monitor the maximum change in messages or assignments between iterations.
  - Set a threshold \( \epsilon \) (e.g., \( 10^{-4} \)) for convergence.

- **Edge Cases**:
  - Ensure handling of cases where messages can become \( -\infty \) or \( \infty \).
  - Implement checks to prevent numerical overflow or underflow.

---

### **Summary of Steps**

1. **Initialize Messages**: Set \( \alpha_{ij}^l = 0 \) and \( \phi_j^0 = 0 \).
2. **Iteratively Update Messages**:
   - Compute adjusted preferences \( \hat{c}_j^l \).
   - Update \( \rho_{ij}^l \) and \( \alpha_{ij}^l \).
   - Compute upward messages \( \tau_j^{l+1} \).
   - Compute downward messages \( \phi_j^l \).
3. **Check for Convergence**: If messages have converged, proceed; else, repeat step 2.
4. **Extract Assignments**: Determine exemplars \( e_j^l \) and assignments \( h_{ij}^l \).
5. **Output**: The hierarchical clustering structure defined by \( h_{ij}^l \) and \( e_j^l \).

---

By explicitly defining all variables, functions, indices, and ranges, and carefully specifying the equations with correct indexing, this guide should help you implement the factor graph for HAP without introducing off-by-one errors or ambiguities.
"""

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



class MaxSumInference:
    def __init__(self, factor_graph):
        self.graph = factor_graph
        self.messages = {}  # Messages between variables and factors

    def initialize_messages(self):
        # Initialize messages to small random values
        self.messages = {}
        for factor in self.graph.factors:
            for var in factor.variables:
                # Messages from factor to variable
                self.messages[(factor, var)] = np.random.uniform(-0.1, 0.1, len(var.domain))
                # Messages from variable to factor
                self.messages[(var, factor)] = np.random.uniform(-0.1, 0.1, len(var.domain))

    def run_max_sum(self, max_iters=100, convergence_threshold=1e-5, damping=0.5):
        self.initialize_messages()
        for iteration in range(max_iters):
            max_diff = 0.0  # Track the maximum message change for convergence

            # Update messages from variables to factors
            for var in self.graph.variables:
                for factor in var.neighbors:
                    # Compute the message from variable to factor
                    incoming_messages = [
                        self.messages[(f, var)] for f in var.neighbors if f != factor
                    ]
                    new_message = self.compute_var_to_factor_message(var, incoming_messages)
                    # Apply damping
                    old_message = self.messages[(var, factor)]
                    damped_message = damping * new_message + (1 - damping) * old_message
                    # Update message and track maximum change
                    max_diff = max(max_diff, np.max(np.abs(damped_message - old_message)))
                    self.messages[(var, factor)] = damped_message

            # Update messages from factors to variables
            for factor in self.graph.factors:
                for var in factor.variables:
                    # Compute the message from factor to variable
                    incoming_messages = [
                        self.messages[(v, factor)] for v in factor.variables if v != var
                    ]
                    new_message = self.compute_factor_to_var_message(factor, var, incoming_messages)
                    # Apply damping
                    old_message = self.messages[(factor, var)]
                    damped_message = damping * new_message + (1 - damping) * old_message
                    # Update message and track maximum change
                    max_diff = max(max_diff, np.max(np.abs(damped_message - old_message)))
                    self.messages[(factor, var)] = damped_message

            # Check for convergence
            if max_diff < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break

        if iteration == max_iters - 1:
            print(f"Did not converge after {max_iters} iterations")

    def compute_var_to_factor_message(self, var, incoming_messages):
        # Max-sum: Sum incoming messages (since log-domain)
        message = np.sum(incoming_messages, axis=0)
        return message

    def compute_factor_to_var_message(self, factor, var, incoming_messages):
        # Max-sum: Compute the max over assignments
        other_vars = [v for v in factor.variables if v != var]
        other_domains = [v.domain for v in other_vars]
        var_domain = var.domain

        # Initialize message
        message = np.full(len(var_domain), -np.inf)
        
        # Debug information
        print(f"Computing message for factor {factor.name} to variable {var.name}")
        print(f"Factor variables: {[v.name for v in factor.variables]}")
        print(f"Other variables: {[v.name for v in other_vars]}")
        
        # Iterate over possible values of the variable
        for i, val in enumerate(var_domain):
            max_value = -np.inf
            # Sum factor potential and incoming messages
            for assignments in np.ndindex(*[len(d) for d in other_domains]):
                assignment = {v.name: v.domain[idx] for v, idx in zip(other_vars, assignments)}
                assignment[var.name] = val
                
                # Debug: Print current assignment
                print(f"Current assignment: {assignment}")
                print(f"Factor name: {factor.name}")
                print(f"Variable name: {var.name}")
                print(f"Factor variables: {[v.name for v in factor.variables]}")
                
                try:
                    # Compute factor potential
                    potential = factor.potential_func(assignment)
                    print(f"Computed potential: {potential}")
                    # Sum incoming messages
                    total = potential + sum(
                        [msg[idx] for msg, idx in zip(incoming_messages, assignments)]
                    )
                    if total > max_value:
                        max_value = total
                except KeyError as e:
                    print(f"KeyError in factor {factor.name}: {str(e)}")
                    print(f"Assignment causing error: {assignment}")
                    print(f"Factor variables: {[v.name for v in factor.variables]}")
                    raise  # Re-raise the exception after printing debug info
            message[i] = max_value
        return message

    def compute_beliefs(self):
        # Compute max-marginal beliefs for variables
        beliefs = {}
        for var in self.graph.variables:
            incoming_messages = [self.messages[(factor, var)] for factor in var.neighbors]
            belief = sum(incoming_messages)
            beliefs[var.name] = belief
        return beliefs


# Now, implement the HAP algorithm using the above classes
def hap_clustering(s, c, L, max_iters=100, convergence_threshold=1e-5, damping=0.5):
    """
    Hierarchical Affinity Propagation clustering algorithm.

    :param s: 3D numpy array of similarities of shape (L, N, N)
              s[l, i, j] is the similarity between data point i and j at layer l
    :param c: 2D numpy array of preferences of shape (L, N)
              c[l, j] is the preference for data point j to be an exemplar at layer l
    :param L: Total number of layers
    :param max_iters: Maximum number of iterations for message passing
    :param convergence_threshold: Convergence threshold for message passing
    :return: Assignments of data points to exemplars at each layer
    """
    N = s.shape[1]  # Number of data points

    # Initialize variables and factors
    variables = {}
    factors = []

    # Create variables h_{ij}^l and e_j^l for all i, j, l
    for l in range(1, L + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                var_name = f"h_{i}_{j}_{l}"
                variables[var_name] = Variable(var_name, [0, 1])  # Binary variable
            var_name = f"e_{i}_{l}"
            variables[var_name] = Variable(var_name, [0, 1])  # Binary variable

    # Create the factor graph
    fg = FactorGraph()

    # Add variables to the factor graph
    for var in variables.values():
        fg.add_variable(var)

    # Define potential functions for factors
    # Similarity function S_{ij}^l(h_{ij}^l)
    def similarity_potential(assignment, s_value):
        if 'h' not in assignment:
            print(f"Error: 'h' not found in assignment. Assignment keys: {assignment.keys()}")
            return -np.inf  # or some other appropriate default value
        h_value = assignment['h']
        return s_value * h_value

    # Exemplar preference function C_j^l(e_j^l)
    def preference_potential(assignment, c_value, var_name):
        if var_name not in assignment:
            print(f"Error: '{var_name}' not found in assignment. Assignment keys: {assignment.keys()}")
            return -np.inf  # or some other appropriate default value
        e_value = assignment[var_name]
        return c_value * e_value

    # Assignment constraint function I_i^l
    def assignment_constraint_potential(assignment, e_prev_value=None, l=None, i=None):
        print(f"Debug: assignment_constraint_potential called for layer {l}, data point {i}")
        print(f"Debug: assignment: {assignment}")
        print(f"Debug: e_prev_value: {e_prev_value}")
        
        # Sum over h_{ij}^l for j in 1..N
        h_keys = [key for key in assignment if key.startswith(f'h_{i}_')]
        print(f"Debug: h_keys: {h_keys}")
        
        h_sum = sum(assignment[key] for key in h_keys)
        print(f"Debug: h_sum: {h_sum}")
        
        if l == 1:
            # Layer 1: h_sum should be close to 1
            result = -abs(h_sum - 1)
        else:
            # Layers l > 1: h_sum should be close to e_i^{l-1}
            if e_prev_value is None:
                print(f"Error: e_prev_value is None for layer {l}, data point {i}")
                result = -np.inf
            else:
                result = -abs(h_sum - e_prev_value)
        
        print(f"Debug: Returning result: {result}")
        return result

    # Exemplar consistency function E_j^l
    def exemplar_consistency_potential(assignment):
        e_value = assignment['e']
        j = assignment['j']
        h_jj_key = f"h_{j}_{j}_{assignment['l']}"  # Add layer information
        if h_jj_key not in assignment:
            print(f"Error: {h_jj_key} not found in assignment")
            return -np.inf
        h_jj = assignment[h_jj_key]
        max_h_ij = max(assignment.get(f"h_{i}_{j}_{assignment['l']}", 0) for i in range(1, N + 1))
        if e_value == h_jj == max_h_ij:
            return 0
        else:
            return -np.inf

    # Build factors and connect them to variables
    for l in range(1, L + 1):
        # Similarity and preference factors
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                # Similarity factor S_{ij}^l
                var_h = variables[f"h_{i}_{j}_{l}"]
                s_value = s[l - 1, i - 1, j - 1]
                def potential_func(assignment, s_val=s_value, var_name=var_h.name):
                    return similarity_potential({'h': assignment[var_name]}, s_val)
                factor = Factor(
                    f"S_{i}_{j}_{l}", [var_h], potential_func
                )
                fg.add_factor(factor)

            # Exemplar preference factor C_j^l
            var_e = variables[f"e_{i}_{l}"]
            c_value = c[l - 1, i - 1]
            def potential_func(assignment, c_val=c_value, var_name=var_e.name):
                return preference_potential(assignment, c_val, var_name)
            factor = Factor(
                f"C_{i}_{l}", [var_e], potential_func
            )
            fg.add_factor(factor)

    # Assignment constraint factors I_i^l
    for l in range(1, L + 1):
        for i in range(1, N + 1):
            h_vars = [variables[f"h_{i}_{j}_{l}"] for j in range(1, N + 1)]
            if l == 1:
                # Layer 1: No e_i^{l-1}
                def potential_func(assignment, l=l, i=i):
                    return assignment_constraint_potential(assignment, l=l, i=i)
                factor = Factor(
                    f"I_{i}_{l}", h_vars, potential_func
                )
            else:
                # Layers l > 1: Include e_i^{l-1}
                var_e_prev = variables[f"e_{i}_{l - 1}"]
                all_vars = h_vars + [var_e_prev]
                def potential_func(assignment, l=l, i=i, var_e_prev=var_e_prev):
                    e_prev_value = assignment.get(var_e_prev.name)
                    if e_prev_value is None:
                        print(f"Error: e_prev_value is None for layer {l}, data point {i}")
                        print(f"Debug: assignment keys: {assignment.keys()}")
                        print(f"Debug: var_e_prev.name: {var_e_prev.name}")
                    return assignment_constraint_potential(assignment, e_prev_value, l=l, i=i)
                factor = Factor(
                    f"I_{i}_{l}", all_vars, potential_func
                )
            fg.add_factor(factor)

    # Exemplar consistency factors E_j^l
    for l in range(1, L + 1):
        for j in range(1, N + 1):
            h_vars = [variables[f"h_{i}_{j}_{l}"] for i in range(1, N + 1)]
            var_e = variables[f"e_{j}_{l}"]
            all_vars = h_vars + [var_e]
            def potential_func(assignment, j=j, l=l, all_vars=all_vars, var_e=var_e):
                print(f"Debug: Exemplar consistency potential for j={j}, l={l}")
                print(f"Debug: Assignment keys: {assignment.keys()}")
                print(f"Debug: All vars: {[var.name for var in all_vars]}")
                try:
                    assignment_with_j = {var.name: assignment[var.name] for var in all_vars}
                    assignment_with_j['j'] = j
                    assignment_with_j['e'] = assignment[var_e.name]
                    return exemplar_consistency_potential(assignment_with_j)
                except KeyError as e:
                    print(f"KeyError in exemplar consistency potential: {str(e)}")
                    print(f"Assignment: {assignment}")
                    return -np.inf
            factor = Factor(
                f"E_{j}_{l}", all_vars, potential_func
            )
            fg.add_factor(factor)

    # Run max-sum inference
    inference = MaxSumInference(fg)
    inference.run_max_sum(max_iters=max_iters, convergence_threshold=convergence_threshold, damping=damping)

    # Extract beliefs
    beliefs = inference.compute_beliefs()

    # Determine assignments based on beliefs
    assignments = {}
    for l in range(1, L + 1):
        e_values = np.zeros(N, dtype=int)
        h_values = np.zeros((N, N), dtype=int)
        for i in range(1, N + 1):
            # Determine exemplar indicator e_j^l
            var_e = variables[f"e_{i}_{l}"]
            belief = beliefs[var_e.name]
            e_values[i - 1] = var_e.domain[np.argmax(belief)]

            # Determine assignments h_{ij}^l
            for j in range(1, N + 1):
                var_h = variables[f"h_{i}_{j}_{l}"]
                belief = beliefs[var_h.name]
                h_values[i - 1, j - 1] = var_h.domain[np.argmax(belief)]
        assignments[l] = {
            'e': e_values,
            'h': h_values
        }

    return assignments

# Example usage
if __name__ == "__main__":
    try:
        # Example data: Similarity matrix and preferences using create_block_matrix
        N = 10  # Number of data points (should be even)
        L = 2  # Number of layers

        def create_block_matrix(L, N):
            # creates a block matrix [[A,0],[0,B]] where A and B are NxN matrices
            n = N // 2
            A = np.random.rand(L, n, n)
            B = np.random.rand(L, n, n)

            sim = np.block([[A, np.zeros((L, n, n))], [np.zeros((L, n, n)), B]])
            preferences = np.random.rand(L, N)

            return sim, preferences

        s, c = create_block_matrix(L, N)

        # Run HAP clustering
        assignments = hap_clustering(s, c, L)

        # Print assignments and debug information
        for l in range(1, L + 1):
            print(f"Layer {l}:")
            e_values = assignments[l]['e']
            h_values = assignments[l]['h']
            print(f"Exemplars (e): {e_values}")
            print(f"Assignments (h):\n{h_values}")
            
            # Debug: Print all variable names
            print("\nDebug: Variable names")
            for var_name in assignments[l].keys():
                if var_name.startswith('h_'):
                    print(var_name)
            print("\n")

        # Visualize the results
        import matplotlib.pyplot as plt

        for l in range(1, L + 1):
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(s[l-1], cmap='viridis')
            plt.title(f'Similarity Matrix (Layer {l})')
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(assignments[l]['h'], cmap='viridis')
            plt.title(f'Assignments (Layer {l})')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()




