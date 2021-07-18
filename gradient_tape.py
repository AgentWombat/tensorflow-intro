import tensorflow as tf

# Define some variables

print("--SIMPLE GRADIENT TAPE EXAMPLE--")
x = tf.Variable([-4,-3,-2,-1,0,1,2,3,4], dtype = 'float32')
print("x =", x)

# tf.GradientTape allows us to track our arithmetic operations, allowing
# us to calculates gradients (derivatives) with very few lines of code.

# Everything inside this 'with' statement is tracked by the GradientTape.
with tf.GradientTape() as tape:
    y = x**2

print("y = x^2 =", y)

# This is how we can get the derivatives of 'y' with respect to
# 'x'.
dy_dx = tape.gradient(y, x)
print("dy/dx =", dy_dx)
# For each value of 'x', we now have a corresponding derivative value.
# As a reminder, the derivative of function 'f' evaluated at point 'l'
# tells us how we expect the value of 'f' to change if 'l' is increased.
# In this case, if we increase the first entree in 'x', -4, we would
# expect the corrisponding 'y' value to drop--for example (-4)^2 > (-3)^2.
# The derivative tells us this by having a negative value corresponding to
# x = -4.

# One interpretation of these value are that each gives the direction of
# steepest ascent. That is, to maximize the increase of value 'y', each
# entree of 'x' should be updated by adding a positive multiple of its
# corresponding derivative value to itself. The effect in this situation
# would be incresing the negative elements of 'x', leaving zero the same,
# and increasing the positive elements of 'x'.
# To minimize 'y', then, we should do the oposite: we should update each
# entree of 'x' by subtracting a positive multiple of the corresponding
# derivative value.

# Let us utilize this for machine learning.
print("--MACHINE LEARNING--")
# Here, we will create an example dataset.
# (it is values for x and y such that y = 4*x + 5):

# The odd syntax here creates an array with numbers 0-99, inclusive.
x = tf.Variable([i for i in range(100)], dtype = 'float32')
y = 4*x + 5

print("The first 10 elements of x:", x[:10])
print("The first 10 elements of y:", y[:10])

# We will use machine learning to find the best values for 'w' and 'b' in
# the formula "y = w*x + b".
# That is, without telling the computer that y = 4*x + 5, it will figure
# that out through machine learning. By the end, we hopefully get values
# for 'w' and 'b' which are close to '4' and '5'.

# Make our initial guesses for 'w' and 'b'
w = tf.Variable(1.0)
b = tf.Variable(1.)

# We here define our "call" function. This is how we get output from our
# model.
def call(x, w, b):
    return w*x + b

# Next is our loss function, which evaluates the models performance.
# We will use MSE (Mean squared error).
def MSE(y, y_hat):
    return tf.reduce_mean((y-y_hat)**2)

# Here is our training loop. We will update our parameters 1000 times
for i in range(1000):
    
    # Tack our calculatiobs
    with tf.GradientTape() as tape:
        y_hat = call(x,w,b)
        loss = MSE(y, y_hat)
    
    # Get our derivatives
    gradients = tape.gradient(loss, [w,b])
    
    # Unpack gradients into seperate variables.
    # 'dw' represents the derivative of the loss with respect to w
    # and 'db' represents the derivative of the loss with respect to b. 
    dw = gradients[0]
    db = gradients[1]

    # Update parameters
    # We multiple by a small number because it ensures that we do not
    # explode 'w' and 'b'.
    learning_rate = 0.000001
    w.assign(w - dw * learning_rate)
    b.assign(b - db * learning_rate)

print("Through machine learning, the computer found w to be", w)
print("Through machine learning, the computer found b to be", b)




