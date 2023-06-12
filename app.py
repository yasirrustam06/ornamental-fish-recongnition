from tensorflow.keras.preprocessing.image import load_img
from skimage.feature import greycomatrix, greycoprops
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from skimage.measure import shannon_entropy
from skimage.filters import sobel
import tensorflow as tf
import numpy as np 
import pandas as pd
import pickle
import cv2
app = Flask(__name__)


# Loading your trained model.......
my_model = load_model('FishRecoModeL.h5')
# these are the Classses names which we want to classify and recongnize.....
class_labels = {0:'Black Sea Sprat',1:'Hourse Mackerel',2:'Red Mullet',3:'Shrimp',4:'Striped Red Mullet'}


## This one is the Feature Extractor Function........
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  # iterate through each file
        # print(image)

        df = pd.DataFrame()  # Temporary data frame to capture information for each loop.
        # Reset dataframe to blank after each loop.

        img = dataset[image, :, :]
        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        # Full image
        # GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = greycomatrix(img, [1], [0])
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr

        GLCM2 = greycomatrix(img, [3], [0])
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        GLCM3 = greycomatrix(img, [5], [0])
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3

        GLCM4 = greycomatrix(img, [0], [np.pi / 4])
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4

        GLCM5 = greycomatrix(img, [0], [np.pi / 2])
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5

        image_dataset = image_dataset.append(df)
    return image_dataset




def Recongnize_image(img_path):
    fish_image = load_img(img_path, target_size=(224, 224))
    fish_array = np.array((fish_image))
    fish_gray = cv2.cvtColor(fish_array, cv2.COLOR_BGR2GRAY)
    biateral_image = cv2.bilateralFilter(fish_gray,15,75,75)
    fish_ex = np.expand_dims(biateral_image, 0)
    img_features = feature_extractor(fish_ex)
    Image_array = np.array(img_features)
    reshaped_Image = Image_array.reshape(5, 5)
    res_arr = np.resize(reshaped_Image, (100, 100))
    Flatten_Image = np.expand_dims(res_arr, 0)
    prediction = np.argmax(my_model.predict(Flatten_Image))
    Image_Class = class_labels[prediction]
    
    return  Image_Class


# routes

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route('/about')
def about():
	return render_template("about.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		result = Recongnize_image(img_path)

	return render_template("index.html", prediction = result, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)