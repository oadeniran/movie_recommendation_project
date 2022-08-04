import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn.neighbors import NearestNeighbors
import pickle

app = Flask(__name__)
app.secret_key = 'ros@y-popping'
with open('titles.pickle', 'rb') as f:
    title_df = pickle.load(f)
with open('knn.pickle', 'rb') as f:
    knn = pickle.load(f)
with open('knn2.pickle', 'rb') as f:
    knn2 = pickle.load(f)
with open('train_rating.pickle','rb') as f:
    train_rating = pickle.load(f)
with open('train_rating_only.pickle','rb') as f:
    train_rating_only  = pickle.load(f)



@app.route('/', methods = ['GET', 'POST'])
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/title_list/', methods = ['POST'])
def list_title():
    title = request.form['moviename'].split(' ')
    if len(title) == 1:
        error = None
        if len(title[0]) <= 3:
            error = 'Word is too short'
            return render_template('index.html', error = error)
        else:
            flash('Succesful')
            titles = title_df[title_df['primaryTitle'].str.contains(title[0])].primaryTitle
    else:
        titles = title_df.copy()
        for word in title:
            if word != 'of':
                word = word.capitalize()
            titles = pd.DataFrame(titles[titles['primaryTitle'].str.contains(word)].primaryTitle)
        titles = titles['primaryTitle']

    return render_template('titles_list.html', titles = titles)

@app.route('/prediction', methods = ['GET','POST'])
def prediction():
    if request.method == 'POST':
        selected = request.form['selected']
        print(selected)
    else:
        selected = session['formdata']
        print(selected)
    
    id = title_df[title_df['primaryTitle'] == selected].index[0]
    dist, n = knn.kneighbors(np.array(train_rating.loc[id, :]).reshape(1,-1))
    dist2, n2 = knn2.kneighbors(np.array(train_rating_only.loc[id, :]).reshape(1,-1))
    titles = title_df.loc[n[0], 'primaryTitle'].values
    titles2 = title_df.loc[n2[0], 'primaryTitle'].values
    final_l = list(titles) + list(titles2)
    return render_template('reccomendations.html', titles = final_l)

@app.route('/repredict', methods = ['GET','POST'])
def repredict():
    session['formdata'] = request.form.get('selected_again')
    return redirect(url_for('prediction'), code = 302)
    

if __name__ == "__main__":
    app.run(debug = False)