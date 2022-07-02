import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
@app.route('/')
def home():
    list1 = ['a','b']
    return render_template('index.html', var = list1)

if __name__ == "__main__":
    app.run()