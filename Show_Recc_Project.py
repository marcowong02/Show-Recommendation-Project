from flask import Flask, render_template, request
import numpy as np          # For math functions
import pandas as pd         # For data processing
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def getDF(): 
    path = 'C:\\Users\\marco\\OneDrive\\Desktop\\Random Project\\animes.csv'
    df = pd.read_csv(path)
    return df

def fixGenre(genre):
    genre = eval(genre)     # list
    genreString = ' '.join(genre)
    return genreString

def work(text):
    df = getDF()
    vectorizer = TfidfVectorizer()
    df['fixed_genre'] = df['genre'].apply(fixGenre)

    featureVectors = vectorizer.fit_transform(df['fixed_genre']) # applies TfidfVectorizer with each respective shows' genres
    # shape of (19311, 47)
    # higher value means more significant, more unique compared to total count
    genre = vectorizer.transform([text])
    cosSimilarity = cosine_similarity(genre.toarray(), featureVectors.toarray()).flatten() # returns how similar the genre is to each and every other genre/feature of shows
    # has to be first parameter of cos similarity, second is what we compare against
    sortedShows = np.argsort(-cosSimilarity) # returns array of indces from smallest to largest
    firstNSorted = sortedShows[0:10]
    shows = df.iloc[firstNSorted]
    shows = shows.sort_values("score", ascending=False)
    shows = shows.drop(columns = ['fixed_genre','aired', 'episodes', 'uid', 'members', 'popularity', 'ranked','img_url','link'])
    # shows.reset_index(drop=True, inplace=True)
    shows.set_index('title', inplace=True)

    return shows.to_html(classes='table table-striped')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/display', methods=['POST'])
def display():
    animeName = request.form['animeName']
    shows = work(animeName)
    return f'{shows}'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)