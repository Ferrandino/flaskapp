# Eye For Art 

This repo contains the flask app for Eye For Art, an art recommendation web app (www.eyeforart.xyz). The app streamlines the process of finding art to pair with items you already own. You simply upload an image of a piece of art in your home and Eye For Art recommends pieces for art from minted.com that share similar features. You know have everything you need to put together your own gallary wall. 

## How it works

The app loads a retrained VGG16 model as well as a feature array for each image in the database. The user image is then run through the model and mapped onto feature space with the database images. The items closest to the image are selected as the top recommendations for your personalized gallery wall. You can then link to each item to buy. 

The app is run via flaskapp/laskexample/views.py. The functions for selecting pieces of art are located in functions_new.py. Saved models and feature arrays are located in flaskapp/flaskexample.models. The database of art images are located in flaskapp/flaskexample/static/class_art. Templates for the webpage are located in flaskapp/flaskexample/templates. The input.html allows the user to upload the image. The recommendations.html returns the recommended images and links. 

## Where did the data come from

Data was scrapped from Minted.com with Beautiful Soup. Items were grouped based on similar features shared in each piece of art. This data was hand labeled and used to retrain the VGG16 image classification model.






 
