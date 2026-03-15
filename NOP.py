import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
def convex_function(x):
    return x**2 + 4*x + 4
def convex_gradient(x):
    return 2*x + 4
def convex_hessian(x):
    return 2
def nonconvex_function(x):
    return x**4 - 3*x**3 + 2
def nonconvex_gradient(x):
    return 4*x**3 - 9*x**2
def nonconvex_hessian(x):
    return 12*x**2 - 18*x
def gradient_descent(grad_func, x0, lr=0.1, iterations=50):

    x = x0
    history = []

    for i in range(iterations):
        gradient = grad_func(x)
        x = x - lr * gradient
        history.append(x)

    return x, history
def newton_method(grad_func, hessian_func, x0, iterations=20):

    x = x0
    history = []

    for i in range(iterations):
        gradient = grad_func(x)
        hessian = hessian_func(x)

        x = x - gradient/hessian
        history.append(x)

    return x, history
x0 = 10

start = time.time()
gd_result_convex, gd_history_convex = gradient_descent(convex_gradient, x0, 0.1, 30)
gd_time = time.time() - start

start = time.time()
newton_result_convex, newton_history_convex = newton_method(convex_gradient, convex_hessian, x0, 10)
newton_time = time.time() - start

print("Convex Function Optimization")
print("Gradient Descent Result:", gd_result_convex)
print("Newton Method Result:", newton_result_convex)
start = time.time()
gd_result_nonconvex, gd_history_nonconvex = gradient_descent(nonconvex_gradient, 5, 0.01, 40)
gd_nc_time = time.time() - start

start = time.time()
newton_result_nonconvex, newton_history_nonconvex = newton_method(nonconvex_gradient, nonconvex_hessian, 5, 15)
newton_nc_time = time.time() - start

print("\nNon Convex Function Optimization")
print("Gradient Descent Result:", gd_result_nonconvex)
print("Newton Method Result:", newton_result_nonconvex)
x = np.linspace(-10,10,200)
y = convex_function(x)

plt.plot(x,y)
plt.title("Convex Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
x = np.linspace(-2,4,200)
y = nonconvex_function(x)

plt.plot(x,y)
plt.title("Non Convex Function")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.show()
plt.plot(gd_history_convex, label="Gradient Descent")
plt.plot(newton_history_convex, label="Newton Method")

plt.xlabel("Iterations")
plt.ylabel("x value")
plt.title("Convex Function Convergence")

plt.legend()
plt.show()
plt.plot(gd_history_nonconvex, label="Gradient Descent")
plt.plot(newton_history_nonconvex, label="Newton Method")

plt.xlabel("Iterations")
plt.ylabel("x value")
plt.title("Non Convex Function Convergence")

plt.legend()
plt.show()
results = pd.DataFrame({

"Method":["Gradient Descent","Newton Method","Gradient Descent","Newton Method"],

"Function":["Convex","Convex","Non Convex","Non Convex"],

"Iterations":[len(gd_history_convex),
len(newton_history_convex),
len(gd_history_nonconvex),
len(newton_history_nonconvex)],

"Final Value":[gd_result_convex,
newton_result_convex,
gd_result_nonconvex,
newton_result_nonconvex],

"Execution Time":[gd_time,
newton_time,
gd_nc_time,
newton_nc_time]

})

print(results)
# Future Work 1: Multi-dimensional Gradient Descent

def multi_function(x):
    return x[0]**2 + x[1]**2 + 4*x[0] + 2*x[1]

def multi_gradient(x):
    dx = 2*x[0] + 4
    dy = 2*x[1] + 2
    return np.array([dx,dy])


def gradient_descent_multi(x0, lr=0.1, iterations=50):

    x = np.array(x0)
    history = []

    for i in range(iterations):

        grad = multi_gradient(x)
        x = x - lr * grad

        history.append(x.copy())

    return x, history


result_multi, hist_multi = gradient_descent_multi([5,5])

print("Optimal point (multi-variable):", result_multi)
# Future Work 2: Linear Regression using Gradient Descent

# Generate dataset
np.random.seed(0)

X = np.linspace(0,10,100)
y = 3*X + np.random.randn(100)


# Initialize parameters
w = 0
b = 0

lr = 0.001
iterations = 500


loss_history = []

for i in range(iterations):

    y_pred = w*X + b

    error = y_pred - y

    dw = (2/len(X)) * np.sum(error*X)
    db = (2/len(X)) * np.sum(error)

    w = w - lr*dw
    b = b - lr*db

    loss = np.mean(error**2)
    loss_history.append(loss)

print("Trained weight:", w)
print("Trained bias:", b)
plt.plot(loss_history)

plt.title("Loss Convergence for Linear Regression")

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.show()