import os
import copy
import time
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from model import generator, discriminator


SAMPLES = 7
BATCHSZ = 3
DIMENSN = 80
NEPOCHS = 50000
PRCPATH = "Data"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#Data preparation
volumes = []
def normaliseImage(I):
	coeff = I - np.min(I)
	numer = 1 - (-1)
	denom = np.max(I) - np.min(I)
	if denom == 0: return (-1)
	else: return coeff * (numer / denom) + (-1)
def cropImage(I, corner, dims):
	c = copy.copy(corner)
	for i in range(3):
		if c[i] < 0:
			pad_elem = [(0, 0), (0, 0), (0, 0)]
			pad_elem[i] = (-c[i], dims[i] + c[i] - I.shape[i])
			pad_elem = tuple(pad_elem)
			I = np.pad(I, pad_elem, 'constant', constant_values=0)
			c[i] = 0
	d, h, w = dims
	z, y, x = c
	return I[z:z+d, y:y+h, x:x+w]
for imgIDX in range(SAMPLES):
	imgPTH = PRCPATH + "/img_" + str(imgIDX) + "_0.nii.gz"
	imgSTK = sitk.GetArrayFromImage(sitk.ReadImage(imgPTH))
	imgSTK = normaliseImage(imgSTK)
	imgSTK = cropImage(imgSTK, [DIMENSN//2,DIMENSN//2,DIMENSN//2], [DIMENSN,DIMENSN,DIMENSN])
	volumes.append(imgSTK)
volumes = np.asarray(volumes)
volumes = volumes.reshape(volumes.shape[0], DIMENSN, DIMENSN, DIMENSN, 1).astype('float32')
train_datasets = tf.data.Dataset.from_tensor_slices(volumes).shuffle(SAMPLES).batch(BATCHSZ)

#Models
genr = generator()
disc = discriminator()

#Optimisers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)
def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	return real_loss + fake_loss
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
dscrmntor_optimizer = tf.keras.optimizers.Adam(1e-4)

#Checkpoint
checkpoint_dir = './Checkpoints3D'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=genr, discriminator=disc)

#Train
@tf.function
def train_step(images):
	noise = tf.random.normal([BATCHSZ, 100])
	with tf.GradientTape() as genr_tape, tf.GradientTape() as disc_tape:
		genr_output = genr(noise, training=True)
		real_output = disc(images, training=True)
		fake_output = disc(genr_output, training=True)
		genr_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)
	gradients_of_generator = genr_tape.gradient(genr_loss, genr.trainable_variables)
	gradients_of_dscrmntor = disc_tape.gradient(disc_loss, disc.trainable_variables)
	generator_optimizer.apply_gradients(zip(gradients_of_generator, genr.trainable_variables))
	dscrmntor_optimizer.apply_gradients(zip(gradients_of_dscrmntor, disc.trainable_variables))
	return genr_loss, disc_loss
#
example = tf.random.normal([1, 100])
for epoch in range(NEPOCHS):
	start = time.time()
	for image_batch in train_datasets:
		gloss, dloss = train_step(image_batch)
	print ('Loss for epoch {} is {:.2f} and {:.2f}'.format(epoch + 1, gloss, dloss))
	#Save the model
	if (epoch + 1) % 1000 == 0:
		checkpoint.save(file_prefix=checkpoint_prefix)
		predictions = genr(example, training=False)
		sitk.WriteImage(sitk.GetImageFromArray(predictions[0,:,:,:,0]), "Epochs/image_{:04d}.nii.gz".format(epoch+1))
