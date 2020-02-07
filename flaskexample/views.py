from flask import render_template
from flask import request
from flaskexample import app
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import urllib.request
import requests
from flask import send_from_directory, flash, redirect
from functions_new import *
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2
UPLOAD_FOLDER = '/home/ubuntu/flaskapp-master/flaskexample/static/data/upload_folder/'
UPLOAD_POST_FOLDER = 'static/data/upload_folder/'
IMAGE_FOLDER ='static/class_art/'

recom_path = '/home/ubuntu/flaskapp-master/flaskexample/static/data/art_for_app.csv'
pd.set_option('display.max_colwidth', 1000)
art = pd.read_csv(recom_path, index_col = 0)


#app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_POST_FOLDER'] = UPLOAD_POST_FOLDER
app.config['IMAGE_DIR'] = IMAGE_FOLDER
app.config.update(
    TESTING=True,
    SECRET_KEY=b'secret_key'
)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('input.html')


@app.route('/recommendations', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('input.html')

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print('No file 1')
            return render_template('input.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            print('No file 2')
            return render_template('input.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #file saved to upload folder
            #load model            
            #get image ready
            model = get_model()            
            processedimage = read_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #detect feature of new image
            feature_dict,size = extract_img_features(model,os.path.join(app.config['UPLOAD_FOLDER'], filename),processedimage)
            #calc lsh
            lsh = calc_LSH(feature_dict,size)
            #return 3 closest things
            F,F1,F2,F3 = get_similar_item(os.path.join(app.config['UPLOAD_FOLDER'], filename), feature_dict, lsh, 3)
            #return the links
            F1_base, F2_base, F3_base, x1, x2, x3 = get_link(F1,F2,F3)

            #group ouput
#	    panel_img = [F1_base, F2_base, F3_base]
            panel_img = [F1_base,F2_base,F3_base]
            print(panel_img)
            panel_link = [x1, x2, x3]
            print(panel_link)
#            print(panel_link)
#            print(panel_link)

            linktext = "Select"
            original = os.path.join(app.config['UPLOAD_POST_FOLDER'], filename)
            panel_img_path = [os.path.join(app.config['IMAGE_DIR'],x) for x in panel_img]
            print(original)
            print(panel_img_path)
            returnval = render_template('recommendations.html', panel_img_path = panel_img_path, original = original, panel_link = panel_link, linktext = linktext, piclink = art['link'].iloc[0])
           
        return returnval
    
            
            
    return render_template('input.html')

#if __name__ == "__main__":
#	app.run(debug=True)
#def go():
#    query = request.args.get('query', '')
#    return render_template(
#        'recommendations.html',
#        query=query,
#    )
