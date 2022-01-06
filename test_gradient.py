import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = tf.Variable(6.0)
dy_dx = g.gradient(y, x)

print(dy_dx)