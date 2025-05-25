
import string 
from gensim.models import Word2Vec
import PyPDF2

def extraer_texto_pdf(ruta):
    with open(ruta,'rb') as archivo:
        lector = PyPDF2.PdfReader(archivo)
        texto = ""
        for pagina in range(len(lector.pages)):
            texto+= lector.pages[pagina].extract_text()

        return texto
    
documento = extraer_texto_pdf("C:/Users/legom/Documents/spacyy/nlp/libros_historiaMX.pdf")

#dividir el documento usando , como separador 

oraciones=documento.split('.')

oraciones_limpias=[]

for oracion in oraciones:
    #eliminar puntuacion
    tokens=oracion.translate(str.maketrans('',

                                    '',string.punctuation)).split()
    
    #convertir a minusculas

    tokens=[word.lower() for word in tokens if word.isalpha()]
    if tokens:#añadir solo si hay tokens
        oraciones_limpias.append(tokens)


#entrenar con el modelo word2vec
model=Word2Vec(sentences=oraciones_limpias,
               vector_size=500,window=5,min_count=1,workers=10)

palabras_cercanas= model.wv.most_similar("México",topn=10)
print(palabras_cercanas)
#guardar el modelo en txt
model.wv.save_word2vec_format('historiamexico.txt',binary=False)
