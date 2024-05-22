import os
from openai import OpenAI
from credentials.credentials import OPENAI_API_KEY

client = OpenAI()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_answer(question : str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content