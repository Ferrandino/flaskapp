from flask import Flask
app = Flask(__name__,static_folder = '/home/ubuntu/flaskapp-master/flaskexample/static')

from flaskexample import views
