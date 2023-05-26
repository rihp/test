from flask import Flask, request, jsonify
app = Flask(__name__)
import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DeepLake

load_dotenv()

os.environ.get("ACTIVELOOP_TOKEN")
username = "rihp" # replace with your username from app.activeloop.ai
projectname = "polywrap5" # replace with your project name from app.activeloop.ai

embeddings = OpenAIEmbeddings(disallowed_special=())

db = DeepLake(dataset_path=f"hub://{username}/{projectname}", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

model = ChatOpenAI(model_name='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


@app.route('/ask', methods=['POST'])
def ask_question():
    print('hello there')
    data = request.get_json()
    prompt = data['question']

    questions = [prompt]
    chat_history = []

    for question in questions:
        prefix = """Use this code snippet as a base example. If you miss any require argument, leave it empty.
            const result = await client.invoke({
            uri: "wrap/ens",
            method: "registerDomain",
            args: {
                domain: "amiguillo.eth",
                owner: "",
                registrarAddress: "",
                connection: {
                networkNameOrChainId: network
                }
            }"""
        sufix = """From the schema.graphql every element should be implemented if it's marked with an exclamation point, like 'String!' implemented argument. """
        result = qa({"question": f'{prefix} {question} {sufix}', "chat_history": chat_history})
        chat_history.append((f'{prefix} {question} {sufix}', result['answer']))
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