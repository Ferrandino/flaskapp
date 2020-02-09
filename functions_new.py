# basic functions & analytics

import os
import numpy as np
import pandas as pd
import ntpath

# making predictions & extracting features with model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras.backend import clear_session
import tensorflow as tf

# calculating image similarity
# calculating image similarity
from lshash_2.lshash_2 import LSHash
from tqdm import notebook
import matplotlib.pyplot as plt
from PIL import Image

# pinging SQL
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import psycopg2

# loading classified image features
import pickle
#-------------------------------------------------------------------------------

mainpath = '/home/ubuntu/flaskapp-master/flaskexample'
user_image_path = '/home/ubuntu/flaskapp-master/flaskexample/static/data/upload_folder/'
CSV = '/home/ubuntu/flaskapp-master/flaskexample/static/data/art_for_app.csv'


#os.getcwd()
#modelfile = os.getcwd()+"/models/VGG_cat.smallerLR.h5"
#clear_session()
#model = load_model(modelfile)


def get_model():
    global model
    model = load_model('VGG_cat.smallerLR.h5')
    print('* Model loaded!')
    return model


def read_image(file):
    img = image.load_img(file,target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    processedimage = preprocess_input(x)
    return processedimage
    
print("* Loading Keras model...")
get_model()
graph = tf.get_default_graph()

#def extract_img_features(model,upload_image_path,processedimage):
#    feature_dict = pickle.load(open(mainpath+"/models/VGG_cat_layer3.p","rb"))
#    get_layer = K.function([model.input],[model.layers[3].output])
#    newimgfeature = get_layer([processedimage])[0].flatten()
#    feature_dict[upload_image_path] = newimgfeature
#    size = np.shape(newimgfeature)
#    return feature_dict,size

def extract_img_features(model,upload_image_path,processedimage):
    feature_dict = pickle.load(open(mainpath+"/models/VGG_cat_layer3.p","rb"))
    get_layer = K.function([model.input],[model.layers[3].output])
    newimgfeature = get_layer([processedimage])[0].flatten()
    feature_dict[upload_image_path] = newimgfeature
    size = np.shape(newimgfeature)
    return feature_dict,size


def calc_LSH(feature_dict, size):
    featuresize = np.shape(feature_dict)

    # params
    k = 6 # hash size
    L = 6  # number of tables
    d = size[0] # Dimension of Feature vector
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)

    # LSH on all the images
    for img_path, vec in notebook.tqdm(feature_dict.items()):
        lsh.index(vec, extra_data=img_path)
    
    return lsh

def get_similar_item(file, feature_dict, lsh_variable, n_items):
    idx =list(feature_dict.keys()).index(file)
    response = lsh_variable.query(feature_dict[list(feature_dict.keys())[idx]].flatten(),
                 num_results=n_items+1, distance_func='l1norm')
    F = response[0][0][1]
    F1 = response[1-2][0][1]
    F2 = response[1-3][0][1]
    F3 = response[1-4][0][1]
    return F,F1,F2,F3
    
def find_price_url(F1):

    dbname = 'art'
    username = 'postgres' # change this to your username
    pw = 'postgres'

    db = create_engine('postgres://%s:%s@localhost/%s'%(username,pw,dbname))
#    print(engine.url)
    con = None
    con = psycopg2.connect(database = dbname, user = username, host='localhost',  password = pw)

    listing = []
    # query:
    sql_query = """
    SELECT link FROM art_table WHERE item=F1_base;
    """
    listing.append(pd.read_sql_query(sql_query,con))

    listing = pd.concat(listing)
    #listing['listing_id'] = similar_listings[1::]

    return listing


def get_link(F1,F2,F3):
    pd.set_option('display.max_colwidth', 1000)
    df = pd.read_csv(CSV,index_col = 0)
    F1_base = ntpath.basename(F1)
    F2_base = ntpath.basename(F2)
    F3_base = ntpath.basename(F3)
    x1 = df.loc[df['item'] == F1_base, 'link'].iloc[0]
    x2 = df.loc[df['item'] == F2_base, 'link'].iloc[0]
    x3 = df.loc[df['item'] == F3_base, 'link'].iloc[0]

    return F1_base, F2_base, F3_base, x1, x2, x3
    
