import numpy as np
from flask import Flask, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/home',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #print(model)
    return render_template('home.html', prediction_text='Movies {} '.format(model))

@app.route("/about")
def about():
    return render_template("about.html")
    
if __name__ == "__main__":
    app.run(debug=True)