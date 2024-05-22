
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the CSV file
data = pd.read_csv('sample.csv')


# Extract the input and output values
x_values = data.iloc[:, 0]
y_values = data.iloc[:, 1]


# loss function
def compute_loss(x_v, y_v, a, b):
    total_loss = 0.0
    n = len(x_v)
    for i in range(n):
        y_pred = a * x_v[i] + b
        total_loss += (y_v[i] - y_pred) ** 2
    return total_loss / n


# Gradient Descent
def gradient_descent(x_values, y_values, learning_rate, iterations=1000):
    a_gradient = b_gradient = 0
    n = len(x_values)
    data = {'a': [], 'b': [], 'loss': [], 'iteration': []}
    gradient_descent_df = pd.DataFrame(data)
    for i in range(iterations):
        try:
            y_pred = a_gradient * x_values + b_gradient
            loss = ((1 / n) * sum([val ** 2 for val in (y_values - y_pred)]))
            md = -(2 / n) * sum(x_values * (y_values - y_pred))
            bd = -(2 / n) * sum(y_values - y_pred)
            a_gradient = a_gradient - learning_rate * md
            b_gradient = b_gradient - learning_rate * bd
            gradient_descent_df.loc[i] = [a_gradient, b_gradient, loss, i]
        except OverflowError as error:
            print(error)
            return gradient_descent_df
    return gradient_descent_df


def stochastic_gradient_descent(x_values, y_values, learning_rate, iterations=1000):
    a_gradient = b_gradient = 0
    data = {'a': [], 'b': [], 'loss': [], 'iteration': []}
    df = pd.DataFrame(data)
    n = len(x_values)
    for i in range(iterations):
        index = np.random.randint(n)
        y_pred = a_gradient * x_values[index] + b_gradient
        a_gradient = a_gradient - learning_rate * (-2 / n) * x_values[index] * (y_values[index] - y_pred)
        b_gradient = b_gradient - learning_rate * (-2 / n) * (y_values[index] - y_pred)
        loss = compute_loss(x_values, y_values, a_gradient, b_gradient)
        df.loc[i] = [a_gradient, b_gradient, loss, i]

    return df


def mini_batch_gradient_descent(x_values, y_values, L=0.0001,iterations= 1000):
    n = len(x_values)
    a = b = 0
    my_data = pd.DataFrame({'a':[],'b':[],'loss':[],'iteration':[]})
    batch = 10
    for i in range(iterations):
        try:
            Rindex = np.random.randint(0,high = n-batch)
            xi = x_values[Rindex:Rindex+batch]
            yi = y_values[Rindex:Rindex+batch]
            y_predict = a * xi + b
            da = (-2/batch) * sum(xi * (yi - y_predict))
            db = (-2/batch) * sum(yi - y_predict)
            a -= L * da
            b-= L * db
            loss = (1/batch) * sum([f**2 for f in (yi-y_predict)])
            my_data.loc[i] = [a, b,loss, i]
        except:
            print("There is a problem, try to change the learning rate")

            return my_data
    return my_data


# Run the algorithms 


gradient_descent_0001 = gradient_descent(x_values, y_values, 0.0001)
gradient_descent_01 = gradient_descent(x_values, y_values, 0.1)

mini_batch_01 = mini_batch_gradient_descent(x_values, y_values, 0.1)
mini_batch_0001 = mini_batch_gradient_descent(x_values, y_values, 0.0001)


stochastic_0001 = stochastic_gradient_descent(x_values,y_values,0.0001)
stochastic_01 = stochastic_gradient_descent(x_values,y_values,0.1)


fig, ax = plt.subplots(1, 3,figsize = (15,6))
# Plot the three lines
ax[0].plot(gradient_descent_0001['iteration'], gradient_descent_0001['a'])
ax[1].plot(gradient_descent_0001['iteration'], gradient_descent_0001['b'])
ax[2].plot(gradient_descent_0001['iteration'], gradient_descent_0001['loss'])

fig.suptitle('gradient descent 0.0001'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot(gradient_descent_01['iteration'], gradient_descent_01['a'])
ax[1].plot(gradient_descent_01['iteration'], gradient_descent_01['b'])
ax[2].plot(gradient_descent_01['iteration'], gradient_descent_01['loss'])

fig.suptitle('gradient descent 0.1'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()




#stochastic
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot(stochastic_0001['iteration'], stochastic_0001['a'])
ax[1].plot(stochastic_0001['iteration'], stochastic_0001['b'])
ax[2].plot(stochastic_0001['iteration'], stochastic_0001['loss'])

fig.suptitle(' stochastic 0.0001'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot(stochastic_01['iteration'], stochastic_01['a'])
ax[1].plot(stochastic_01['iteration'], stochastic_01['b'])
ax[2].plot(stochastic_01['iteration'], stochastic_01['loss'])

fig.suptitle('  stochastic 0.1'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


# mini bat

fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot(mini_batch_0001['iteration'], mini_batch_0001['a'])
ax[1].plot(mini_batch_0001['iteration'], mini_batch_0001['b'])
ax[2].plot(mini_batch_0001['iteration'], mini_batch_0001['loss'])

fig.suptitle('  mini batch 0.0001 '
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot(mini_batch_01['iteration'], mini_batch_01['a'])
ax[1].plot(mini_batch_01['iteration'], mini_batch_01['b'])
ax[2].plot(mini_batch_01['iteration'], mini_batch_01['loss'])

fig.suptitle('  mini batch 0.1 '
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('a per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('loss per epoch')

#ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("a")
ax[1].set_ylabel("b")
ax[2].set_ylabel("loss")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()

