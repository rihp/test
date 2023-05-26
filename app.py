from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app, allow_headers=["Origin", "X-Requested-With", "Content-Type", "Accept"])


app.config['CORS_HEADERS'] = 'Content-Type'


import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DeepLake

load_dotenv()

os.environ.get("ACTIVELOOP_TOKEN")
username = "rihp" # replace with your username from app.activeloop.ai
projectname = "polywrap11" # replace with your project name from app.activeloop.ai

embeddings = OpenAIEmbeddings(disallowed_special=())

db = DeepLake(dataset_path=f"hub://{username}/{projectname}", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

model = ChatOpenAI(model_name='gpt-4', temperature=0) # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


@app.route('/ask', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def ask_question():
    print('hello there')
    data = request.get_json()
    prompt = data['question']

    questions = [prompt]
    chat_history = []

    for question in questions:
        prefix = ""
        sufix = ". I just want the code structure . Dont give me any additional text or comments. Remember to use double quotes"

        result = qa({"question": f'{prefix} {question} {sufix}', "chat_history": chat_history})
        # chat_history.append((f'{prefix} {question} {sufix}', result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
    

    answer = result['answer']

    return jsonify({"answer": answer})  


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our medium-greeting-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run()