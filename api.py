from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from db import save_message
from main import get_response, predict_class, intents

app = FastAPI()

class ChatInput(BaseModel):
    message: str
    user_id: str = "default"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat/welcome")
async def welcome():
    return {"response": "Hola 👋 Soy tu asistente contable. ¿En qué puedo ayudarte?"}

async def chat_endpoint(input: ChatInput):
    user_id = input.user_id
    message = input.message

    # Predict the intent from the message
    intents_list = predict_class(message)
    
    # Get the appropriate response
    response = get_response(intents_list, intents, user_id)

    # Save the message and response to the database
    save_message(
        user_id=user_id,
        message=message,
        intent=intents_list[0]["intent"],  # You can update this logic to be more specific
        response=response,
        context=None
    )

    return {"response": response}
