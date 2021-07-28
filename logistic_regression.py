import tensorflow as tf
import random
import math
# Logistic regression is a type of model which makes a binary prediction:
# "Yes" or "No"; "Satisfied" or "Not Satisfied"; "Act" or "Don't Act".

# To the computer, each one of these dicisions/outcomes will be
# represented by either a "1" or a "0". For instance, "1" might be
# "Satisfied" and "0" "Not Satisfied".

# For ease of access, we will concoct our own dataset.
# We will pretend that our dataset contains examples in which people's
# height and stamina are logged alongside their respective
# statis as either (0) "Not a Basketball Player" or (1) "Basketball
# Player".
print("--DATA--")
x = []
y = []
# Create 100 examples
for i in range(100):
    
    # "random.random()" generates a floating-point number between 0 and 1.
    # we use this to generate random heights between 0 and 1. '0' meaning
    # short and '1' meaning huge!

    height = random.random()
    
    # Likewise, we use "random.random()" to get stamina scores.
    stamina = random.random()
    
    # We pack the height and weight into one variable and save that in
    # the "x" data.
    stats = [height, stamina]
    x.append(stats)

    # For simplicity, if someone's height + stamina - 1 > 0,
    # we say that the person plays basketball. Else, he does not.
    bball_score = height + stamina - 1

    if bball_score > 0:
        y.append(1.)
    else:
        y.append(0.0)


print("The first 10 elements of x are", x[:10])
print("The first 10 elements of y are", y[:10])

x = tf.Variable(x)
y = tf.Variable(y)

print("As Tensorflow variables, x[:10] =", x[:10])
print("y[:10] =", y[:10])

# We will now implement a logistic regression model using Tensorflow which
# will predict whether a person is a basketball player or not just by
# knowing his height and his stamina score. The end goal should output '1'
# for a lad which plays basketball and '0' for one who does not.
print("--LOGISTIC_REGRESSION--")

# We could try using plain ol' linear regression, but we would find
# that for this problem, it does not make much. With linear regression,
# the output could range from -infinity to +infinity; here, we want
# all of our outputs to be between 0 and 1.

# What logistic regression does is take the output of linear regression
# and stick it into a sigmoid function.
# "sigmoid(x) = 1 / (1 + e^(-x))"
# For this problem, that means our model's output is given by
# "sigmoid( w1 * height + w2 * stamina + b )"
# This makes it so that all of our models predictions will be within the
# the range of 0-1.

# Note that for all of these functions, 'x' will be a two dimensional
# Tensorflow Variable.

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def call(x, w1, w2, b):
    # x[:, 0] will grab all of the height stats and x[:, 1] will grab all
    # of the the stamina stats. More generally, x[:, C] means in every
    # row, grab only the value in column number C.
    return sigmoid(w1 * x[:,0] + w2 * x[:,1] + b)

# Mean squared error is not a good loss function to use with logistic
# regression. Binary crossentropy is a much better choice. Try both
# and compare the results.
def MSE(y, y_hat):
    return tf.reduce_mean( (y - y_hat)**2)

def binary_crossentropy(y, y_hat):
    
    # Because of floating-point arithmetic, I do this to ensure that
    # no '0.0' or '1.0' is passed into the logarithm.
    y_hat = 0.999*(y_hat + 0.0001)
    
    # See if you can figure out why this works well for binary data.
    # Hint: Values in 'y' are either '1' or '0'.
    loss = -( y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat) )
    
    # Get average (sum all values in 'loss' and divide by number of values)
    loss = tf.reduce_mean(loss)
    return loss

# Make our guesses for our parameters
w1  = tf.Variable(random.random())
w2 = tf.Variable(random.random())
b = tf.Variable(random.random())

for i in range(1000):
    
    # Make predictions and calculate loss
    with tf.GradientTape() as tape:
        y_hat = call(x, w1, w2, b)
        loss = binary_crossentropy(y, y_hat)
    
    # Get gradients
    gradients = tape.gradient(loss, [w1, w2, b])
    
    dw1 = gradients[0]
    dw2 = gradients[1]
    db = gradients[2]
    
    # Update parameters with gradients
    learning_rate = 0.1

    w1.assign(w1 - dw1 * learning_rate)
    w2.assign(w2 - dw2 * learning_rate)
    b.assign(b - db * learning_rate)
    

print("w1 , w2 , b =",w1.numpy(),",", w2.numpy(),",", b.numpy())

# Here, we will get the accuracy of our model.
y_hat = call(x,w1,w2,b)

num_correct = 0
for y_i, y_i_hat in zip(y.numpy(), y_hat.numpy()):
    if abs(y_i - y_i_hat) < 0.5:
        num_correct += 1

accuracy = num_correct / y.shape[0]

print("MODEL ACCURACY: " + str(accuracy *100) + "%")
