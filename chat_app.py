
from flask import Flask, render_template, request, session, redirect, url_for
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import uuid
import os

# ----------------------------------------------------
                # FLASK APP
# ----------------------------------------------------

app = Flask(__name__)
app.secret_key = "my_flask_secret_123"


# ----------------------------------------------------
                # LOAD LLM MODEL
# ----------------------------------------------------

API_KEY = os.getenv("GROQ_API_KEY")

llm_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=API_KEY,
    temperature=0.5
)


# ----------------------------------------------------
                # MEMORY OBJECT
# ----------------------------------------------------

conversations = {}
chat_histories = {}


# ----------------------------------------------------
                # CHAT MEMORY
# ----------------------------------------------------

def get_conversation():

    if "chat_id" not in session:
        session["chat_id"] = str(uuid.uuid4())

    chat_id = session["chat_id"]

    if chat_id not in conversations:

        buffer_memory = ConversationBufferMemory(
            return_messages=True
        )

        conversations[chat_id] = ConversationChain(
            memory=buffer_memory,
            llm=llm_model,
            verbose=False
        )

        chat_histories[chat_id] = []

    return chat_id, conversations[chat_id]


# ----------------------------------------------------
                # HOME PAGE
# ----------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    chat_id, conversation = get_conversation()

    if request.method == "POST":

        qn = request.form.get("question", "").strip()

        if qn:

            if qn.lower() in ["stop", "done"]:
                return redirect(url_for("reset"))

            # Store Human Message
            chat_histories[chat_id].append({
                "role": "user",
                "text": qn
            })

            # Get AI Response
            try:
                response1 = conversation.predict(input=qn)
                chat_histories[chat_id].append({
                    "role": "assistant",
                    "text": response1
                    })
            except Exception as e:
                print("CHATBOT ERROR:", repr(e))
                return f"Chatbot Error: {e}", 500

            # Store AI Response
            chat_histories[chat_id].append({
                "role": "assistant",
                "text": response1
            })

    return render_template(
        "index.html",
        messages=chat_histories.get(chat_id, [])
    )


# ----------------------------------------------------
                # RESET CHAT
# ----------------------------------------------------

@app.route("/reset")
def reset():

    chat_id = session.get("chat_id")

    if chat_id:
        conversations.pop(chat_id, None)
        chat_histories.pop(chat_id, None)

    session.pop("chat_id", None)

    return redirect(url_for("home"))


# ----------------------------------------------------
                # RUN FLASK APP
# ----------------------------------------------------

if __name__=='__main__':  
    app.run(host='0.0.0.0', port = '8000', debug = True ) 
