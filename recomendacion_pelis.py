import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
df = pd.read_csv('C:/Users/legom/Documents/spacyy/nlp/movie_metadata.csv')

df['genres']=df['genres'].str.replace('|',' ')
df['plot_keywords']=df['plot_keywords'].str.replace('|',' ')

#hacer una nueva columna juntando generos y plot keywords
df['texto']=df[['genres','plot_keywords']].apply(lambda row: ' '.join(row.values.astype(str)),axis=1)
'''
imprimir
row=df[['genres','plot_keywords','texto']].iloc[0]
print(row)
'''

tfidf = TfidfVectorizer(max_features=10000)
x=tfidf.fit_transform(df['texto'])

peliculas = pd.Series(df.index, index=df['movie_title'])
peliculas.index=peliculas.index.str.strip()

pelicula1=peliculas['ZMD: Zombies of Mass Destruction']
pelicula2=peliculas['I Love You, Don\'t Touch Me!']
def calcular_similitud(indice):
    similitud=cosine_similarity(x[indice],x)
    similitud=similitud.flatten()
    return (-similitud).argsort()[1:11]

print(df['movie_title'].iloc[calcular_similitud(pelicula1)])
print(df['movie_title'].iloc[calcular_similitud(pelicula2)])
print(calcular_similitud(pelicula1),calcular_similitud(pelicula2))