# [ON-GOING] Linear Programming Solver

## Code Architecture

### Files hierarchy

```mermaid
graph TD; 
  subgraph "lp_module"
    P[param.py] --> B[basis.py]
    PS([_primal_simplex.py])
    DS([_dual_simplex.py])
    B --> S[slp.py]
    PS -.- S
    DS -.- S
    S --> F[lp.py]
    F --> MILP[milp.py]
    BB[_branch_and_bound.py] -.- MILP
    MILP --> I[[__init__.py]]
    
  end
  I -.-> M[main.py]
  subgraph "unit_tests"
    V[simplex_validation.py]
  end
  I -.-> V
```
### Class diagram

```mermaid
---
config:
  htmlLabels: false
---
classDiagram
    Animal <|-- Duck
    note for Duck "can fly<br>can swim<br>can dive<br>can help in debugging"
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Basis{
        • int n
        • int m
        • np.ndarray B
        • np.ndarray N
        • np.ndarray x
        • np.ndarray y
        • spla.SuperLU lu_solver
        
        update_lu()
        extract_baseII()
    }
    class Constraint{
        • np.ndarray **a**
        • str **symbol**
        • float **b_l**
        • float **b_u**
    }
    
    class LinearProblem{
        • int **n**
        • int **m** 
        • list[Constraint] **constraints**
        • np.ndarray **x_l**
        • np.ndarray **x_u**
        • np.ndarray **c**
        • flag_max **bool**
        
        set_objective()
        set_variable_bounds()
        set_constraints()
        from_mps() LinearProblem$
        to_SLP()
        solve()
        getResult()
    }
    
    class MILP_Problem{
        • np.ndarray **int_vars**
        set_variables_bounds()
        check_integrity()
        solve_relaxation()
        solve
    }
    
    class SLP_Model{
        • sparse.csc_matrix **A**
        • np.ndarray **b**
        • np.ndarray **c**
        • int **n**
        • int **m**
        • float **offset**
        • np.ndarray **col_scales**
        scale_model()
        modelPhaseI
        restraint()

    }
    
    LinearProblem <|-- MILP_Problem
    LinearProblem *-- Constraint
    
```

## Hypotheses 
- Variables must be lower bounded.

## Usage : example of a knapsack problem

We show here how to solve a basic MILP problem with our module. The following knaspack example will be taken as example. 
```math
\begin{align}
\max \hspace{0.5cm}& 8x_1 + 11x_2 + 6x_3 + 4x_4 \\
\text{s.t.   }\hspace{0.5cm} & 5x_1 + 7x_2 + 4x_3 + 3x_4 \leq 14
\end{align}
```


### Problem definition

- **0) Loading module and creating a MILP instance**
```python
from milp_module import *
import numpy as np
milp = MILP_Problem() # creation of a MILP instance
```

- **1) Defining the objective**:
```python
c = np.array([8., 11., 6., 4.]) # objective coefficients
milp.set_objective("Max",c) # only "Min" or "Max" are allowed
```
- **2) Defining bounds and integer conditions of the variables**:
```python
x_l = np.array([0., 0., 0., 0.])
x_u = np.array([1., 1., 1., 1.])
int_vars = np.array([True, True, True, True])
milp.set_variable_bounds(x_l,x_u,int_vars)
```
- **3) Defining constraints**

\[Note\]: the method `set_constraints` remove the potential old constraints stored in the `MILP_Problem` instance.
```python
a1 = np.array([5., 7., 4., 3.])
b1_l = -np.inf
b1_u = 14.
cstr1 = Constraint(a1,"<=",b1_l,b1_u)
milp.set_constraints([cstr1])
```

### Solving the problem
You may choose the level of logging you want with the parameter `verbosity`. The default value `-1` prints nothing. You can choose the safety for the maximum number of nodes (sub problems) computed with the parameter `max_nb_nodes`. Various tolerance parameters relative to digital approximation of zero in the Branch & Bound and simplex methods can be set in the [*milp_module/param.py*](milp_module/param.py) file.
```python
x_opti, z_opti = ks.solve(verbosity=0)
```
Once you have run all the lines above, you will get the following result:

```terminal
Max  8.0•X_0 + 11.0•X_1 + 6.0•X_2 + 4.0•X_3
subject to :
 ⦿ -inf <=  5.0•X_0 + 7.0•X_1 + 4.0•X_2 + 3.0•X_3 <= 14.0
 ⦿ X_0 ∈ [0.0, 1.0]
 ⦿ X_1 ∈ [0.0, 1.0]
 ⦿ X_2 ∈ [0.0, 1.0]
 ⦿ X_3 ∈ [0.0, 1.0]

MILS successfully solved with B&B : 	 • z = 20.999999988279633
	 • x = [0. 1. 1. 1.]
```


## References

LP instances were found in the [Netlib dataset](https://www.netlib.org/lp/data/). The files were decompressed thanks to the `emps.c` tool, also found on this page.
