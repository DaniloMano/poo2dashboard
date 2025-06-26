import chromadb
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")


user_query = input("What do you want to know about growing vegetables?\n\n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

print(results['documents'])
#print(results['metadatas'])

# Cria cliente Groq com a chave do .env
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""

#print(system_prompt)

# Envia para o modelo da Groq (LLaMA 3)
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)