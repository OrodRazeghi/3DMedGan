import os


SAMPLES = 7
NFRAMES = 30
RESIZES = 160
PRCPATH = "Data"
NIIPATH = "../CNNRegistration/Datasets/MRI/C"


#preparing data
for samIDX in range(SAMPLES):
	for timIDX in range(NFRAMES):
		imgPTH = NIIPATH + str(samIDX) + "/dcm-" + str(timIDX) + ".nii"
		prcPTH = PRCPATH + "/img_" + str(samIDX) + "_" + str(timIDX) + ".nii.gz"
		commnd = "cp " + imgPTH + " " + prcPTH
		os.system(commnd)
		#
		prcPTH = "img_" + str(samIDX) + "_" + str(timIDX) + ".nii.gz"
		docker = "sudo docker run --rm --volume=/data/or15/GANetwork/Data:/data biomedia/mirtk resample-image"
		dflags = "-imsize " + str(RESIZES) + " " + str(RESIZES) + " " + str(RESIZES) + " -size 1 1 1"
		commnd = docker + " " + prcPTH + " " + prcPTH + " " + dflags
		os.system(commnd)
		#
		prcPTH = "img_" + str(samIDX) + "_" + str(timIDX) + ".nii.gz"
		docker = "sudo docker run --rm --volume=/data/or15/GANetwork/Data:/data biomedia/mirtk edit-image"
		dflags = "-reset"
		commnd = docker + " " + prcPTH + " " + prcPTH + " " + dflags
		os.system(commnd)
		#
		prcPTH = "img_" + str(samIDX) + "_" + str(timIDX) + ".nii.gz"
		docker = "sudo docker run --rm --volume=/data/or15/GANetwork/Data:/data biomedia/mirtk edit-image"
		dflags = "-origin " + str(-RESIZES//2+.5) + " " + str(-RESIZES//2+.5) + " " + str(RESIZES//2-.5) + " -orientation -1 0 0 0 -1 0 0 0 1"
		commnd = docker + " " + prcPTH + " " + prcPTH + " " + dflags
		os.system(commnd)
		print ('Resizing Image {} and Slice {}'.format(samIDX, timIDX))
