import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for, flash
from sklearn.neighbors import NearestNeighbors
import pickle
import gc

app = Flask(__name__)
app.secret_key = 'ros@y-popping'



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
            with open('titles.pickle', 'rb') as f:
                title_df = pickle.load(f)
                titles = title_df[title_df['primaryTitle'].str.contains(title[0])].primaryTitle
            del(title_df)
            gc.collect()
    else:
        with open('titles.pickle', 'rb') as f:
            title_df = pickle.load(f)
        titles = title_df.copy()
        del(title_df)
        for word in title:
            if word != 'of':
                word = word.capitalize()
            titles = pd.DataFrame(titles[titles['primaryTitle'].str.contains(word)].primaryTitle)
        titles = titles['primaryTitle']
        gc.collect()

    return render_template('titles_list.html', titles = titles)

@app.route('/prediction', methods = ['GET','POST'])
def prediction():
    if request.method == 'POST':
        selected = request.form['selected']
        print(selected)
    else:
        selected = session['formdata']
        print(selected)

    with open('titles.pickle', 'rb') as f:
        title_df = pickle.load(f)
    id = title_df[title_df['primaryTitle'] == selected].index[0]

    with open('train_rating.pickle', 'rb') as f:
        train_rating = pickle.load(f)
        with open('knn.pickle', 'rb') as f2:
            knn = pickle.load(f2)
            dist, n = knn.kneighbors(np.array(train_rating.loc[id, :]).reshape(1,-1))
            del(knn)
        del(train_rating)
    with open('train_rating_only.pickle', 'rb') as f:
        train_rating_only= pickle.load(f)
        with open('knn2.pickle', 'rb') as f2:
            knn2 = pickle.load(f2)
            dist2, n2 = knn2.kneighbors(np.array(train_rating_only.loc[id, :]).reshape(1,-1))
            del(knn2)
        del(train_rating_only)

    titles = list(title_df.loc[n[0], 'primaryTitle'].values)
    titles2 = list(title_df.loc[n2[0], 'primaryTitle'].values)
    del(title_df)
    final_l = titles + titles2
    del(titles)
    del(titles2)
    del(dist)
    del(dist2)
    del(n)
    del(n2)
    gc.collect()
    return render_template('reccomendations.html', titles = final_l)

@app.route('/repredict', methods = ['GET','POST'])
def repredict():
    session['formdata'] = request.form.get('selected_again')
    return redirect(url_for('prediction'), code = 302)
    

if __name__ == "__main__":
    app.run(debug = False)