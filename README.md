# Gradient-Desent-
# Gradient Descent Implementation in Python

This repository contains a Python implementation of various gradient descent algorithms for linear regression. The primary goal is to demonstrate the difference between standard Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent with varying learning rates.

## Files

- `gradient_descent.py`: Main script containing the gradient descent algorithms and visualization code.
- `sample.csv`: A sample CSV file containing the dataset for regression analysis.

## Requirements

To run the code, you'll need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib
```

## Description

### Functions

1. **`compute_loss(x_v, y_v, a, b)`**:
    - Computes the mean squared error loss for the given parameters `a` and `b`.
    - Parameters:
        - `x_v`: Input feature values.
        - `y_v`: Output values.
        - `a`: Slope parameter.
        - `b`: Intercept parameter.
    - Returns:
        - Mean squared error loss.

2. **`gradient_descent(x_values, y_values, learning_rate, iterations=1000)`**:
    - Performs batch gradient descent.
    - Parameters:
        - `x_values`: Input feature values.
        - `y_values`: Output values.
        - `learning_rate`: Learning rate for the gradient descent.
        - `iterations`: Number of iterations.
    - Returns:
        - DataFrame containing `a`, `b`, `loss`, and `iteration` for each epoch.

3. **`stochastic_gradient_descent(x_values, y_values, learning_rate, iterations=1000)`**:
    - Performs stochastic gradient descent.
    - Parameters:
        - `x_values`: Input feature values.
        - `y_values`: Output values.
        - `learning_rate`: Learning rate for the gradient descent.
        - `iterations`: Number of iterations.
    - Returns:
        - DataFrame containing `a`, `b`, `loss`, and `iteration` for each epoch.

4. **`mini_batch_gradient_descent(x_values, y_values, L=0.0001, iterations=1000)`**:
    - Performs mini-batch gradient descent.
    - Parameters:
        - `x_values`: Input feature values.
        - `y_values`: Output values.
        - `L`: Learning rate for the gradient descent.
        - `iterations`: Number of iterations.
    - Returns:
        - DataFrame containing `a`, `b`, `loss`, and `iteration` for each epoch.

### Visualization

The script generates visualizations for each gradient descent algorithm with different learning rates, showing the progression of the slope `a`, intercept `b`, and loss over the iterations.

## Usage

1. **Load the CSV file**:
    ```python
    data = pd.read_csv('sample.csv')
    ```

2. **Extract input and output values**:
    ```python
    x_values = data.iloc[:, 0]
    y_values = data.iloc[:, 1]
    ```

3. **Run the gradient descent algorithms**:
    ```python
    gradient_descent_0001 = gradient_descent(x_values, y_values, 0.0001)
    gradient_descent_01 = gradient_descent(x_values, y_values, 0.1)

    mini_batch_01 = mini_batch_gradient_descent(x_values, y_values, 0.1)
    mini_batch_0001 = mini_batch_gradient_descent(x_values, y_values, 0.0001)

    stochastic_0001 = stochastic_gradient_descent(x_values, y_values, 0.0001)
    stochastic_01 = stochastic_gradient_descent(x_values, y_values, 0.1)
    ```

4. **Generate plots**:
    ```python
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].plot(gradient_descent_0001['iteration'], gradient_descent_0001['a'])
    ax[1].plot(gradient_descent_0001['iteration'], gradient_descent_0001['b'])
    ax[2].plot(gradient_descent_0001['iteration'], gradient_descent_0001['loss'])
    fig.suptitle('Gradient Descent 0.0001', fontsize=25)
    ax[0].set_title('a per epoch')
    ax[1].set_title('b per epoch')
    ax[2].set_title('loss per epoch')
    ax[0].set_ylabel("a")
    ax[1].set_ylabel("b")
    ax[2].set_ylabel("loss")
    for i in ax:
        i.set_xlabel("epoch")
    plt.show()
    ```
