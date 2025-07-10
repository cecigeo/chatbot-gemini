import random
import json
import pickle
import numpy as np
import nltk

from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# WordNetLemmatizer() nos da la raíz (forma base) de las palabras
# que el chatbot puede reconocer. Por ejemplo, para "hunting", "hunter", "hunts" y "hunted", 
# la función lemmatize de la clase WordNetLemmatizer() devolverá "hunt" porque es la palabra raíz.
lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # separar las palabras de los patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) # y las agrego a la lista de palabras

        # asocia los patterns con su tag respectiva
        documents.append(((word_list), intent['tag']))

        # appendear los tags a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# guardando las palabras raiz      
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]

words = sorted(set(words))

# guardar las palabras y clases en archivos binarios
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# necesitamos valores numericos de las palabras
# porque la red neuronal necesita numeros para trabajar
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)


# training = np.array(training)

# train_x = list(training[:, 0])
# train_y = list(training[:, 1])

train_x = []
train_y = []
for bag, output_row in training:
    train_x.append(bag)
    train_y.append(output_row)

train_x = np.array(train_x)
train_y = np.array(train_y)


# crear un modelo de ml Sequential
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),
                activation='softmax'))

# compilar el modelo
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)

# guardar modelo
model.save("chatbot.h5", hist)

print("Entrenadoo")