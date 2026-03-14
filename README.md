# [ON-GOING] Linear Programming Solver

## Code Architecture

```mermaid
graph TD; 
  subgraph "lp_module"
    P[param.py] --> B[basis.py]
    B --> S[primal_simplex.py]
    S --> F[lp_formulation.py]
    F --> I[__init__.py]
  end
  I -.-> M[main.py]
```
## References

LP instances were found on [John Burkardt](https://www.researchgate.net/profile/John-Burkardt) educational page via [this link](https://people.sc.fsu.edu/~jburkardt/datasets/mps/).