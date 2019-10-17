import tensorflow as tf
from tensorflow.keras import layers


DIMENSN = 80


def generator():

	skernel = (4,4,4)
	sstride = (2,2,2)
	model = tf.keras.Sequential()
	CDIMNSION = DIMENSN//(2**4)
	#Project and reshape
	model.add(layers.Dense(CDIMNSION*CDIMNSION*CDIMNSION*512, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Reshape((CDIMNSION,CDIMNSION,CDIMNSION,512)))
	#3D convs
	model.add(layers.Conv3DTranspose(256, skernel, strides=(1,1,1), padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv3DTranspose(128, skernel, strides=sstride, padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv3DTranspose( 64, skernel, strides=sstride, padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv3DTranspose( 32, skernel, strides=sstride, padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	#Final layer
	model.add(layers.Conv3DTranspose(  1, skernel, strides=sstride, padding='same', use_bias=False, activation='tanh'))
	return model

def discriminator():

	skernel = (4,4,4)
	sstride = (2,2,2)
	model = tf.keras.Sequential()
	#3D convs
	model.add(layers.Conv3D( 32, skernel, strides=sstride, padding='same', input_shape=[DIMENSN,DIMENSN,DIMENSN,1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	model.add(layers.Conv3D( 64, skernel, strides=sstride, padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	model.add(layers.Conv3D(128, skernel, strides=sstride, padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	model.add(layers.Conv3D(256, skernel, strides=sstride, padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))
	#Final layer
	model.add(layers.Flatten())
	model.add(layers.Dense(1))
	return model
