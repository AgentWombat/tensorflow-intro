import tensorflow as tf

# Tensorflow is a massive machine learning library with tons
# of ready to use datasets, machine-learning functions, and
# upper level APIs.
# At its core, and this is what we will focus on here, Tensorflow
# provides usefull datatypes for data manipulation in machine learning
# alongside automatic differentiation (very nice!).

# When working with Tensorflow, our go-to datatype for storing numbers
# is "Variable".
# We can define scalars (single numbers) like so:
a = tf.Variable(3.0)
b = tf.Variable(5.)
# We include the ".0" or "." to ensure that the variable saves as type
# "float32".
print("--DEFINING SCALARS--")
print("a =", a)
print("b =", b)

# If we every want to get a more typical value out of a Tensorflow variable,
# we use tf.Variable.numpy().
print("a =", a.numpy())
print("b =", b.numpy())

# These variables can be manipulated in a fassion similar to Python variables.
print("a + b =", a+b)

c = 2*a + b / a
print("c = 2 * a + b/a =", c)

# In addition to holding scalars (single numbers), Tensorflow Variables can
# also hold arrays.
print("--DEFINING ARRAYS--")
A = tf.Variable([1,2,3,4.])
B = tf.Variable([3,3,3,9], dtype = 'float32')
C = tf.Variable([[1.,2,3,4],[10,20,30,40],[5,6,7,8],[50,60,70,80]])
# To ensure that 'A' is storing its data as floating point numbers, I place
# a decimal after one of the entrees. This has the same effect as what I
# did for variable 'B'-- I explicitly defined the variables datatype with
# the dtype' argument.
print("A =", A)
print("B =", B)
print("C =", C)
# When Tensorflow variables hold arrays, arithmetic operations work
# element-wise.
print("A + B =", A + B)
print("A / B =", A - B)
print("A^B =", A ** B)

# There is a feature which Tensorflow shares with Numpy called broadcasting.
# Broadcasting automatically grows one array to match that of another in
# order for an operation to carry out. It is best shown through an example:

# The first example utilizes broadcasting, the second shows what
# broadcasting is actually doing.
print("--BROADCASTING--")
one = tf.Variable(1.)
print("A + 1 =", A + one) # instead of "A + one", "A + 1" would have worked.

ones = tf.Variable([1.0,1,1,1])
print("A + 1 = [1,2,3,4] + [1,1,1,1] =", A + ones)

# Here is another example of broadcasting:
print("C + A =", C + A)
# Above, 'A' is broadcasted to be
# [[1,2,3,4]
#  [1,2,3,4]
#  [1,2,3,4]
#  [1,2,3,4]]

print("--REDUCE_MEAN--")
D = tf.Variable([[1,2,3],[30,10,20],[40,30,20]], dtype = 'float32')
print("D =", D)

# "tf.reduce_mean(H)" will return the average value in variable 'H'.
print("tf.reduce_mean(D) =", tf.reduce_mean(D))

# The 'axis' argument specifies an axis along which the average is calculated.
print("tf.reduce_mean(D, axis = 0) =", tf.reduce_mean(D, axis = 0))
print("tf.reduce_mean(D, axis = 1) =", tf.reduce_mean(D, axis = 1))

# To update the value of a Tensorflow variable, one might be tempted to use
# this: "A = A + 1". This will work in most instances; however, it is
# not memory efficient. This construct does not update the Variable 'A' but
# instead creates a new Variable with the value "A + 1" and then assigns
# 'A' to equal that variable.
# Hence, update a variable like so.
print("--ASSIGN--")
print("A =", A)

A.assign(A + 1)
print('After "A.assign(A+1)", A =', A)
