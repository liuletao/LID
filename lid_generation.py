from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *
import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist, cdist, squareform


model = InceptionV3(weights='imagenet')

#print (model.summary())

def preprocess_data(img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x
	#plt.imshow(img)
	#plt.show()

def load_data():
	img_data_list = []
	path = os.getcwd()
	data_path = os.path.join(path,'image_subspace')
	data_list_dir = os.listdir(data_path)
	for img in data_list_dir:
		img_path_dir = os.path.join(data_path, img)
		img_data_list.append(img_path_dir)
	return img_data_list

def load_data_adv():
	img_data_list = []
	path = os.getcwd()
	data_path = os.path.join(path,'adv_image')
	data_list_dir = os.listdir(data_path)
	for img in data_list_dir:
		img_path_dir = os.path.join(data_path, img)
		img_data_list.append(img_path_dir)
	return img_data_list


def preprocess(img):
	img_resized = image.load_img(img, target_size=(299, 299))
	print (img)
	img_processed = preprocess_data(img_resized)
	return img_processed



def get_softmax_output(x):
	intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer("predictions").output)
	intermediate_output = intermediate_layer_model.predict(x)
	print ("shape", intermediate_output.shape)
	return intermediate_output

def classify(x):
	y = model.predict(x)
	for index, res in enumerate(decode_predictions(y)[0]):
	    print('{}. {}: {:.3f}%'.format(index + 1, res[1], 100 * res[2]))

def batch_process(image_batch):
	soft_list = []
	for img in image_batch:
		img_processed = preprocess(img)
		print ("img_processed.shape", img_processed.shape)
		#classify(img_processed)
		softmax_layer = get_softmax_output(img_processed)
		soft_list.append(softmax_layer)
	return soft_list


def get_lid(norm,adv):
	norm = np.asarray(norm, dtype = np.float32)
	adv = np.asarray(adv, dtype = np.float32)
	norm = np.squeeze(norm, axis = 1)
	adv = np.squeeze(adv, axis = 1)
	print ("norm.shape", norm.shape)
	print ("adv.shape", adv.shape)
	k = 40
	f = lambda v: - k / np.sum(np.log(v/v[-1]))
	a = cdist(adv, norm)
	print ("a", a)
	print ("a.shape", a.shape)
	a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
	print ("a", a)
	print ("a.shape", a.shape)
	a = np.apply_along_axis(f, axis=1, arr=a)
	print ("a", a)
	print ("a.shape", a.shape)


if __name__ == "__main__":
	image_batch = load_data()
	norm = batch_process(image_batch)
	adv_batch = load_data_adv()
	adv = batch_process(adv_batch)
	get_lid(norm,adv)

