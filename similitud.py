
import gensim
vectores= gensim.models.KeyedVectors.load_word2vec_format('C:/Users/legom/Documents/spacyy/nlp/SBW-vectors-300-min5.txt')

def analogia(vec1,vec2,vect3):

    similitud = vectores.most_similar(positive=[vec1,vect3],negative=[vec2])
    print(f'{vec1} es a {vec2} como {similitud[0][0]} es a {vect3}')

analogia('rey','hombres','mujer')
analogia('vaca','leche','huevo')

def palabras_cercanas(v):
    cercanas= vectores.most_similar(positive=[v])
    print(f"palabras cercanas de {v}")
    for word,score in cercanas:
        print("\t%s" %word)

palabras_cercanas('Jugador')
#cambia el contexto al iniciar con mayuscula se refiere al pais 
palabras_cercanas('Chile')
palabras_cercanas('chile')