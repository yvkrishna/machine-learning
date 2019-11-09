import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR);
from numpy import *
celsius=array([-40,-10,0,8,15,22,38],dtype=float);
farienheit=array([-40,14,32,46,59,72,100],dtype=float)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))
history = model.fit(celsius, farienheit, epochs=500, verbose=False)
model.predict([100.0])
from matplotlib.pyplot import *
# plot(model.epochs,model.loss)
print("the weights are {}".format(l0.get_weights()))
