# Solving the `HS071` Optimization Problem with PyTorch Autograd and Cyipopt

This example shows how to solve the [`HS071`](https://cyipopt.readthedocs.io/en/stable/tutorial.html) optimization problem using the `cyipopt` interface. While `cyipopt` requires implementations for the gradient, Jacobian, and Hessian, these are dynamically computed using PyTorch's automatic differentiation tools, reducing the need for manual derivative calculations.

```python
import cyipopt
import numpy as np
import torch

class HS071():
    def __init__(self):
        self.x_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True, dtype=torch.float32)
    
    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        return x[0] * x[3] * torch.sum(x[0:3]) + x[2]
    
    def constraint_function(self, x_tensor):
        return torch.stack([torch.prod(x_tensor), torch.dot(x_tensor, x_tensor)])
    
    def gradient_function(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the gradient of the objective function."""
        grad = torch.autograd.grad(self.objective_function(x), x, create_graph=True)[0]
        return grad
    
    def nabla2g(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Returns the jacobian of the i-th constraint function."""
        grad_g = lambda x: torch.autograd.grad(self.constraint_function(x)[i], x, create_graph=True)[0]
        con_jacobian = torch.autograd.functional.jacobian(grad_g, x)
        return torch.tril(con_jacobian)

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        with torch.no_grad():  # Disable gradient tracking temporarily
            self.x_tensor.copy_(torch.from_numpy(x))
            objective_numpy = self.objective_function(self.x_tensor).numpy()
        
        return objective_numpy

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        with torch.no_grad():
            self.x_tensor.copy_(torch.from_numpy(x))
        grad_tensor = self.gradient_function(self.x_tensor)
        with torch.no_grad():
            grad_numpy = grad_tensor.numpy()
        return grad_numpy

    def constraints(self, x):
        """Returns the constraints using PyTorch."""
        with torch.no_grad():  # Disable gradient tracking temporarily
            self.x_tensor.copy_(torch.from_numpy(x))
            constraints = self.constraint_function(self.x_tensor).numpy()

        return constraints

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        with torch.no_grad():  # Disable gradient tracking temporarily
            self.x_tensor.copy_(torch.from_numpy(x))
            
        return torch.autograd.functional.jacobian(self.constraint_function, self.x_tensor)

    def hessianstructure(self):
        """Returns the row and column indices for non-zero vales of the
        Hessian."""

        # NOTE: The default hessian structure is of a lower triangular matrix,
        # therefore this function is redundant. It is included as an example
        # for structure callback.

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        with torch.no_grad():
            self.x_tensor.copy_(torch.from_numpy(x))

        nabla2f = torch.autograd.functional.jacobian(self.gradient_function, self.x_tensor)
        H = obj_factor * nabla2f.numpy()
        for i in range(2): H += lagrange[i]*self.nabla2g(self.x_tensor, i).numpy()
        row, col = self.hessianstructure()
        return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))


lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

cl = [25.0, 40.0]
cu = [2.0e19, 40.0]

x0 = [1.0, 5.0, 5.0, 1.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(cl),
   problem_obj=HS071(),
   lb=lb,
   ub=ub,
   cl=cl,
   cu=cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)

x, info = nlp.solve(x0)
print(x)
```
