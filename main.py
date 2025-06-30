import random
import json
import pickle
import unicodedata
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from db import get_last_messages
from gemini import query_gemini

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot.h5")

context = {}

def normalize_text(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()

def clean_up_sentences(sentence):
    sentence = normalize_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def is_greeting_or_farewell(text):
    """Detecta si el mensaje es un saludo, despedida o agradecimiento básico"""
    text_normalized = normalize_text(text.strip())
    
    greetings = ["hola", "buenos dias", "buenas tardes", "buenas noches", "buen dia", "buena tarde", "buena noche", "hey", "saludos"]
    farewells = ["chau", "adios", "hasta luego", "nos vemos", "bye", "hasta la vista", "que tengas buen dia", "que andes bien"]
    thanks = ["gracias", "muchas gracias", "te agradezco", "mil gracias", "thank you", "thanks"]
    
    # Verificar si el texto completo o alguna palabra clave está presente
    for greeting in greetings + farewells + thanks:
        if greeting in text_normalized or text_normalized == greeting:
            return True
    
    return False

def infer_contextual_intent(user_input, user_id):
    history = get_last_messages(user_id, limit=1)
    if not history:
        return None
    last_bot_msg = history[0][2].lower()
    user_input_lower = normalize_text(user_input.strip())
    
    if user_input_lower in ["si", "dale", "ok", "por favor", "mostralo", "quiero el formulario"]:
        if "reclasificación" in last_bot_msg or "formulario" in last_bot_msg:
            return "confirm_reclassification"
        if "ticket" in last_bot_msg:
            return "confirm_ticket"
    if user_input_lower in ["no", "mejor no", "deja", "olvidalo", "no hace falta"]:
        if "reclasificación" in last_bot_msg:
            return "cancel_reclassification"
        return "cancel_action"
    return None

def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.45
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_welcome_message():
    welcome_intent = next((i for i in intents["intents"] if i["tag"] == "welcome"), None)
    if welcome_intent:
        return random.choice(welcome_intent["responses"])
    return "¡Hola! ¿En qué puedo ayudarte?"

def get_response(intents_list, intents_json, user_id="default"):
    if not intents_list:
        return "No entendí tu mensaje. ¿Podés reformularlo?"
    
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            if "context_filter" in intent:
                if context.get(user_id) != intent["context_filter"]:
                    continue
            if "context_set" in intent:
                context[user_id] = intent["context_set"]
            if tag.startswith("confirm_") or tag.startswith("cancel_"):
                context[user_id] = None
            return random.choice(intent["responses"])
    
    response = query_gemini(tag)
    return response
