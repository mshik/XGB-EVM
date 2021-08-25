import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Flask utils
from flask_ngrok import run_with_ngrok
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)
run_with_ngrok(app)

STATIC_FOLDER = 'static'
MODELS_FOLDER = 'models'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
total_len=0
np.set_printoptions(suppress=True)


#Utilities
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pickle
import seaborn as sns
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.calibration import calibration_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.random.seed(123)
#Model Development
#Feature extraction
import tensorflow
import tensorflow as tf
tf.random.set_seed(123)
from tensorflow import keras
os.system('pip install efficientnet')
import efficientnet.tfkeras as efn
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
#Classifiers
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#Model Evaluation
from sklearn.metrics import accuracy_score,f1_score

    

def load_models(path, model_name='vgg'):
    #Load model wothout classifier/fully connected layers
    model = keras.models.load_model(path)
    if model_name=='mob':
        model = keras.Model(inputs=model.input,outputs=model.get_layer('dense').output)
    elif model_name=='efn':
        model = keras.Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
    else:
        other_model = Sequential()
        for layer in model.layers[:-2]: # just exclude last layer from copying
            other_model.add(layer)
        model = other_model
    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in model.layers:
        layer.trainable = False
    return model


def extract_features(x_input, model_name='gain'):
    if model_name=='pretrained':
        #Load pretrained models
        # 1. MobileNetV2
        mob_path = os.path.join(MODELS_FOLDER, 'mob.model')
        mob_model = load_models(mob_path,'mob')
        print("MobileNetV2 loaded")
        # 2. VGG16
        vgg_path = os.path.join(MODELS_FOLDER, 'vgg.model')
        vgg_model = load_models(vgg_path)
        print("Vgg16 loaded")
        # 3. EfficientNetB4
        efn_path = os.path.join(MODELS_FOLDER, 'efn.model')
        efn_model = load_models(efn_path,'efn')
        print("EfficientNetB4 loaded")
        #Now, let us use features from loaded models for classifier
        mob_feature_extractor=mob_model.predict(x_input)
        VGG_feature_extractor=vgg_model.predict(x_input)
        efn_feature_extractor=efn_model.predict(x_input)
        mob_features = mob_feature_extractor.reshape(mob_feature_extractor.shape[0], -1)
        VGG_features = VGG_feature_extractor.reshape(VGG_feature_extractor.shape[0], -1)
        efn_features = efn_feature_extractor.reshape(efn_feature_extractor.shape[0], -1)
        # Average of features
        X_cnn = np.mean([mob_features, VGG_features, efn_features], axis=0)
    else:
        #Load GAIN model
        gain_path = os.path.join(MODELS_FOLDER, 'gain.model')
        gain_model = load_models(gain_path)
        #Now, let us use features from loaded models for classifier
        gain_feature_extractor=mob_model.predict(x_input)
        X_cnn = gain_feature_extractor.reshape(gain_feature_extractor.shape[0], -1)
    return X_cnn
    

def fetchData(data_path):
    print(data_path)
    x_input = cv2.imread(data_path)
    # BGR -> RGB
    x_input = cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB)
    x_input = tf.image.resize(x_input, size=[96, 96])
    x_input = x_input/255
    x_input = np.expand_dims(x_input, axis=0)
    return x_input


def predict(model_name, data_path):
    #Extract features from input image
    x_input = fetchData(data_path)
    if model_name=='pretrained':
        x_input = extract_features(x_input, model_name=model_name)
    else:
        x_input = extract_features(x_input)
    #Load classifier
    xgb_path = os.path.join(MODELS_FOLDER, 'XGB_classifier.model')
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax')
    xgb_clf.load_model(xgb_path)
    xgb_proba = xgb_clf.predict_proba(x_input)[:, 1]
    xgb_output={'Diabetic Retinopathy':str("{:.2f}".format(xgb_proba[0]*100))+'%',
            'Glaucoma':str("{:.2f}".format(xgb_proba[1]*100))+'%',
            'Macular Degneration':str("{:.2f}".format(xgb_proba[2]*100))+'%',
            'Normal':str("{:.2f}".format(xgb_proba[3]*100))+'%'}
    return xgb_output


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        # Make prediction
        xgb_output = predict('pretrained', fullname)
        if xgb_output=='':
            result='No output'
        
        result=": "+str(xgb_output)[1:len(str(xgb_output))-1]
        return result
        
    return None


if __name__ == '__main__':
    app.run()
