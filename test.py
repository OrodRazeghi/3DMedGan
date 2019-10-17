import tensorflow as tf
from model import generator


#Models
genr = generator()

#Checkpoint
checkpoint_dir = './Checkpoints3D'
checkpoint = tf.train.Checkpoint(generator=genr)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Generate
example = tf.random.normal([1, 100])
predictions = genr(example, training=False)
