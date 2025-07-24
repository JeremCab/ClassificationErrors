import nlopt
import numpy as np

# Problem dimension
n = 10  # number of decision variables

# Define a nonlinear objective function (to be minimized)
def objective(x, grad):
    if grad.size > 0:
        grad[:] = 2 * x  # gradient of x^2
    return np.sum(x**2)

# Define a nonlinear constraint: e.g., sum(x^3) - 1 <= 0
def nonlinear_constraint(x, grad):
    if grad.size > 0:
        grad[:] = 3 * x**2
    return np.sum(x**3) - 1

# Create optimizer
opt = nlopt.opt(nlopt.LD_MMA, n)  # LD_MMA supports nonlinear inequality constraints

# Set lower and upper bounds (optional)
opt.set_lower_bounds([-10.0]*n)
opt.set_upper_bounds([10.0]*n)

# Set objective function
opt.set_min_objective(objective)

# --- Add 768 linear inequality constraints: A_i x <= b_i ---

# Random A and b for demonstration purposes
np.random.seed(0)
A = np.random.randn(768, n)
b = np.random.randn(768)

# Add each linear constraint individually
for i in range(768):
    def lin_constraint_factory(a_row, b_val):
        return lambda x, grad: (
            np.dot(a_row, x) - b_val
            if grad.size == 0
            else (grad.__setitem__(slice(None), a_row) or np.dot(a_row, x) - b_val)
        )
    opt.add_inequality_constraint(lin_constraint_factory(A[i], b[i]), 1e-8)

# Add the nonlinear constraint
opt.add_inequality_constraint(nonlinear_constraint, 1e-8)

# Set optimization parameters
opt.set_xtol_rel(1e-6)
opt.set_maxeval(1000)

# Initial guess
x0 = np.random.randn(n)

# Run optimization
try:
    x_opt = opt.optimize(x0)
    minf = opt.last_optimum_value()
    print("Optimum x:", x_opt)
    print("Minimum objective value:", minf)
except nlopt.RoundoffLimited as e:
    print("NLopt stopped due to roundoff errors:", str(e))
