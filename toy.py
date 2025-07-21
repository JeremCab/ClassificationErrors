import numpy as np
import cyipopt

class ToyProblem(object):
    def objective(self, x):
        return (x[0] - 1)**2 + (x[1] - 2)**2

    def gradient(self, x):
        return np.array([
            2 * (x[0] - 1),
            2 * (x[1] - 2)
        ], dtype=np.float64)

    def constraints(self, x):
        # Nonlinear inequality constraint: x[0]^2 + x[1] >= 1
        return np.array([x[0]**2 + x[1] - 1], dtype=np.float64)

    def jacobian(self, x):
        # Derivative of constraint: [2*x0, 1]
        return np.array([2 * x[0], 1.0], dtype=np.float64)

    def jacobianstructure(self):
        # One constraint, two variables
        return ([0, 0], [0, 1])

# Variable bounds: [x0 >= 0, x0 <= 1.5], x1 unbounded
x_l = np.array([0.0, -np.inf])
x_u = np.array([1.5, np.inf])

# Constraint bounds: x0^2 + x1 - 1 >= 0 → c(x) ∈ [0, ∞)
cl = np.array([0.0])
cu = np.array([np.inf])

x0 = np.array([0.5, 0.5])  # initial guess

nlp = cyipopt.Problem(
    n=2,
    m=1,
    lb=x_l,
    ub=x_u,
    cl=cl,
    cu=cu,
    problem_obj=ToyProblem()
)

nlp.add_option("print_level", 5)
nlp.add_option("tol", 1e-6)

solution, info = nlp.solve(x0)

print("\n✅ Optimal solution:", solution)
print("Objective value:", info["obj_val"])

