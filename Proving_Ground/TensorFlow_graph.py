import tensorflow as tf
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")
result = a + b

# a node of tensor graph
print(a.graph is tf.get_default_graph())

# the result is also a node of tensor graph
print(result.graph is tf.get_default_graph())