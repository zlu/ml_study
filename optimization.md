# Introduction to Optimization in Data Science

Optimization is a cornerstone of data science, enabling us to fine-tune models by adjusting parameters to minimize or maximize an objective function. This blog explores the key components of optimization problems, their classifications, practical applications, and advanced techniques like gradient descent, stochastic gradient descent, and backpropagation for deep neural networks (DNNs).

## Key Components of Optimization

An optimization problem typically involves:

- **Parameters**: Adjustable values, denoted as $\vartheta$, which can be scalars, vectors, or arrays.
- **Objective Function**: A function $L(\vartheta)$ that quantifies the quality of a parameter configuration, often minimized to find the optimal $\theta^*$:

  $$
  \theta^* = \underset{\theta \in \Theta}{\arg \min} L(\theta)
  $$

- **Constraints**: Limitations on parameters, defining a feasible region. For example, a synthesizer knob might be restricted to a range like [0, 10].

## Types of Optimization Problems

Optimization problems are classified based on the nature of parameters and constraints:

- **Continuous vs. Discrete**: Continuous optimization involves parameters in a continuous space ($\mathbb{R}^n$), leveraging smoothness for efficiency. Discrete optimization deals with discrete parameter sets, like integers, which can be more complex.
- **Constrained vs. Unconstrained**: Constrained optimization imposes limits (e.g., equality $c(\theta) = 0$ or inequality $c(\theta) \leq 0$) on parameters, such as a synthesizer knob’s physical range. Unconstrained optimization allows any parameter configuration, often leading to impractical solutions.
- **Convex vs. Non-Convex**: Convex problems guarantee that any local minimum is the global minimum, simplifying the search. Non-convex problems may have multiple local minima or saddle points, complicating optimization.

## Objective Function and Constraints

The objective function, often called a loss function, measures how well parameters fit the desired outcome. For example, in machine learning, the objective might minimize the difference between predicted and actual values:

$$
L(\theta) = \sum_i \|f(\mathbf{x}_i; \theta) - y_i\|
$$

Constraints define the feasible set. For instance, a box constraint restricts parameters to a range (e.g., $0 < \theta_i < 1$), while equality constraints might force parameters onto a surface, like a unit sphere ($\|\theta\|_2 = 1$).

## Practical Examples

### 1. Geometric Median
The geometric median in $\mathbb{R}^2$ minimizes the sum of distances to a set of points:

$$
L(\theta) = \sum_i \|\theta - \mathbf{x}_i\|_2
$$

This can be solved using SciPy’s `minimize` function with the Nelder-Mead method, which iteratively adjusts a simplex to find the optimal point.

### 2. Linear Least Squares
For problems like line fitting ($y = mx + c$), the objective is to minimize the squared error:

$$
L(\theta) = \sum_i (y_i - mx_i - c)^2
$$

This can be solved directly using the pseudo-inverse:

$$
\mathbf{x}^* = (A^T A)^{-1} A^T \mathbf{y}
$$

For example, given $A = \begin{pmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{pmatrix}$, $b = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, the solution is:

$$
\mathbf{x}^* = \begin{pmatrix} 2 \\ 2 \end{pmatrix}
$$

### 3. Constrained Optimization with Lagrange Multipliers
Consider the least squares problem above with an additional constraint $x_1 + x_2 = 1$. The Lagrangian is defined as:

$$
\mathcal{L}(x_1, x_2, \lambda) = \frac{1}{2} \left( \left( \frac{1}{2} x_1 - 1 \right)^2 + \left( \frac{1}{2} x_2 - 1 \right)^2 \right) + \lambda (x_1 + x_2 - 1)
$$

Taking partial derivatives and setting them to zero:

$$
\frac{\partial \mathcal{L}}{\partial x_1} = \frac{1}{4} (x_1 - 2) + \lambda = 0 \implies x_1 = 2 - 4\lambda
$$

$$
\frac{\partial \mathcal{L}}{\partial x_2} = \frac{1}{4} (x_2 - 2) + \lambda = 0 \implies x_2 = 2 - 4\lambda
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = x_1 + x_2 - 1 = 0
$$

Solving these, we get $\lambda = \frac{3}{8}$, $x_1 = 0.5$, $x_2 = 0.5$.

## Optimization Algorithms

### Direct Methods
- **Linear Least Squares**: Solves convex problems directly using normal equations or pseudo-inverse, ideal for problems with a single global minimum.

### Iterative Methods
- **Grid Search**: Evaluates the objective function across a grid of parameter values. It’s simple but inefficient in high dimensions due to the curse of dimensionality.
- **Random Search**: Randomly samples parameters, avoiding local minima but lacking efficiency.
- **Hill Climbing**: Adjusts parameters incrementally, but can get stuck in local minima.
- **Simulated Annealing**: Extends hill climbing with a temperature schedule, allowing occasional uphill moves to escape local minima.
- **Genetic Algorithms**: Mimic evolution with mutation, selection, and crossover, maintaining a population of solutions to explore the parameter space.

### Convex Optimization
Convex problems benefit from specialized solvers like `scipy.optimize.linprog` for linear programming or `minimize` with methods like `trust-constr` for quadratic programming, ensuring efficient convergence to the global minimum.

## Advanced Optimization for Deep Neural Networks

### Why Not Heuristic Search?
Heuristic methods like random search or genetic algorithms are inefficient for DNNs with billions of parameters. They lack convergence guarantees and struggle with hyperparameter tuning.

### Derivatives and Gradients
For a scalar function $f(x): \mathbb{R}^N \rightarrow \mathbb{R}$, the gradient is:

$$
\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_N} \end{pmatrix}
$$

The Jacobian matrix generalizes this for vector functions, while the Hessian matrix ($\nabla^2 L(\theta)$) captures second-order derivatives, indicating curvature:

$$
\nabla^2 L(\theta) = \begin{pmatrix}
\frac{\partial^2 L}{\partial \theta_1^2} & \cdots & \frac{\partial^2 L}{\partial \theta_1 \partial \theta_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 L}{\partial \theta_n \partial \theta_1} & \cdots & \frac{\partial^2 L}{\partial \theta_n^2}
\end{pmatrix}
$$

Eigenvalues of the Hessian reveal the nature of critical points: positive definite (minimum), negative definite (maximum), mixed signs (saddle point), or semidefinite (plateau/ridge).

### Gradient Descent
Gradient descent updates parameters in the direction of the negative gradient:

$$
\theta^{(i+1)} = \theta^{(i)} - \delta \nabla L(\theta^{(i)})
$$

For the least squares problem with $A = \begin{pmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{pmatrix}$, $b = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, starting from $x^{(0)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, and $\delta = 0.1$, the gradient is $\nabla L(x) = A^T (A x - b)$. Two iterations yield improved parameters.

### Stochastic Gradient Descent (SGD)
SGD uses a single or few training examples per iteration, addressing slow computation for large datasets:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t, x^{(i)})
$$

### Mini-Batch SGD
Mini-batch SGD updates parameters using a small subset of data:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t, B_t)
$$

It balances speed and stability, reducing oscillations compared to SGD.

### Momentum
Momentum reduces oscillations in mini-batch SGD by accumulating past gradients:

$$
v_{t+1} = \gamma v_t + \eta \nabla_\theta L(\theta_t), \quad \theta_{t+1} = \theta_t - v_{t+1}
$$

Typically, $\gamma$ is set between 0.8 and 0.99.

### Adagrad
Adagrad adapts the learning rate based on past gradients:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta L(\theta_t)
$$

where $G_t = \sum_{\tau=1}^t \nabla_\theta L(\theta_\tau)^2$.

### RMSprop
RMSprop modifies Adagrad with an exponentially decaying average of squared gradients, improving performance on non-stationary problems.

### Adam Optimizer
Adam combines momentum and RMSprop, using moving averages of gradients and squared gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L(\theta_t), \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L(\theta_t))^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2}, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Typically, $\beta_1 = 0.9$, $\beta_2 = 0.999$.

### Second-Order Methods
Newton’s method uses the Hessian for faster convergence:

$$
w_{\text{new}} = w - H^{-1} \nabla f(w)
$$

However, computing the Hessian is costly ($O(d^2)$ storage and computation). Newton-Conjugate Gradient (CG) solves linear systems efficiently without direct inversion, converging quadratically near minima.

### Deep Neural Networks (DNNs)
DNNs approximate complex functions by minimizing:

$$
\theta^* = \underset{\theta}{\arg \min} \sum_i \|f(\mathbf{x}_i; \theta) - y_i\|
$$

A DNN consists of layers with linear transformations followed by nonlinear activations (e.g., ReLU, sigmoid).

### Backpropagation
Backpropagation efficiently computes gradients using the chain rule:

$$
\frac{d f}{d x} = \frac{d f}{d b} \times \frac{d b}{d c} \times \cdots \times \frac{d g}{d x}
$$

It adjusts weights layer by layer, enabling DNNs to learn complex patterns in fields like image classification and speech recognition.

### Automatic Differentiation
Automatic differentiation computes exact derivatives for complex functions, as seen in libraries like Autograd and JAX. For example, computing the gradient of $\tanh(x)$ using Autograd yields precise results for optimization.

## Practical Considerations

- **Choose the Right Algorithm**: Use direct methods for least-squares, convex solvers for convex problems, and first-order methods like gradient descent when derivatives are available. For unknown problem types, use zeroth-order methods like simulated annealing.
- **Continuity and Differentiability**: Continuous and differentiable functions are easier to optimize. Non-differentiable functions require careful handling.
- **Convergence**: Convex optimization guarantees global minima, while non-convex methods may find local minima.

## Conclusion

Optimization is critical for solving real-world problems in data science, from fitting simple models to training complex DNNs. By understanding parameters, objective functions, constraints, and advanced techniques like SGD, momentum, and backpropagation, practitioners can tackle a wide range of challenges effectively.

For more details, refer to the original lecture slides or explore SciPy’s optimization tools at [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html).
