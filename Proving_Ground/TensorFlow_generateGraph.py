import tensorflow as tf

# this codes generates two computation graph, each graph define a variables named "V"

g1 = tf.Graph()
with g1.as_default():
    # initialize it's value as zero, notice there are parenthesis after "initializer", this is different from the book
    v = tf.get_variable("v",initializer=tf.zeros_initializer()(shape = [1]))

g2 = tf.Graph()
with g2.as_default():
    # initialize it's value as one
    v = tf.get_variable("v",initializer = tf.ones_initializer()(shape=[1]))

with tf.Session(graph=g1) as sees:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse = True):
        print(sees.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sees:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse = True):
        print(sees.run(tf.get_variable("v")))