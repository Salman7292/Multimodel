
from flask import Flask, render_template, request, session ,redirect,url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import markdown  # Add this import

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-123")


load_dotenv()  # Load .env file if using one

# Updated prompt template with formatting instructions
template = """You are Gemini, an AI assistant developed by Google. Follow these rules:
1. Use Markdown formatting for all responses
2. Code blocks must be enclosed in triple backticks with language specification
3. Use bullet points for lists and **bold** for important terms
4. Headings should use # symbols
5. When explaining technical concepts, include code examples in appropriate languages

Current conversation history:
{history}

Human input: {input}
"""
PROMPT = PromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PROMPT
)

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        selected_model = request.form.get('selected_model', 'Gemini')
        
        if selected_model == 'Gemini':
            try:
                response = conversation.predict(input=user_input)
                # Convert Markdown to HTML
                html_response = markdown.markdown(response, extensions=['fenced_code'])
            except Exception as e:
                html_response = f"<div class='error'>Error: {str(e)}</div>"
        else:
            html_response = "This model is not yet implemented"

        session['chat_history'].append({'type': 'human', 'content': user_input})
        session['chat_history'].append({'type': 'ai', 'content': html_response})
        session.modified = True


    

    return render_template('index.html', chat_history=session['chat_history'])




# chatgpt


# Enhanced prompt template with dynamic response control
template_gpt = """You are ChatGPT-4, an AI assistant developed by Openai.and integrated By Salman Malik in this All in one LLM app and Salman malik is the Student of Softwere Engneering at University of Malakand and he is expert in AI. Follow these rules:
1. IDENTITY: If asked about yourself, respond: "I'm ChatGPT, an AI developed by OpenAi"
2. DETAILS: Only provide detailed explanations when user explicitly uses:
- "explain in detail"
- "elaborate on"
- "expand your answer"
- "describe thoroughly"

Current conversation history:
{history}

Human input: {input}

"""
PROMPT_gpt = PromptTemplate.from_template(template_gpt)

# Initialize Groq model
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",  # Alternative: "llama2-70b-4096"
    max_tokens=200
)

memory_gpt = ConversationBufferMemory()
conversation_gpt = ConversationChain(
    llm=llm,
    memory=memory_gpt,
    prompt=PROMPT_gpt  # <-- Using custom prompt
)


@app.route('/gpt', methods=['GET', 'POST'])
def gpt():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        selected_model = request.form.get('selected_model', 'Gemini')
        
        if selected_model == 'Gemini':
            try:
                response = conversation_gpt.predict(input=user_input)
                # Convert Markdown to HTML
                html_response = markdown.markdown(response, extensions=['fenced_code'])
            except Exception as e:
                html_response = f"<div class='error'>Error: {str(e)}</div>"
        else:
            html_response = "This model is not yet implemented"

        session['chat_history'].append({'type': 'human', 'content': user_input})
        session['chat_history'].append({'type': 'ai', 'content': html_response})
        session.modified = True


    

    return render_template('gpt.html', chat_history=session['chat_history'])



# deep seek


# Enhanced prompt template with dynamic response control
template_DeepSeek = """You are DeepSeek-R1, an AI assistant developed by DeepSeek. Follow these rules:
1. IDENTITY: If asked about yourself, respond in deatil : "I’m DeepSeek-R1, an artificial intelligence assistant created by DeepSeek, a Chinese company focused on AI research and applications."
2. DETAILS: Only provide detailed explanations when user explicitly uses:
- "explain in detail"
- "elaborate on"
- "expand your answer"
- "describe thoroughly"

Current conversation history:
{history}

Human input: {input}

:"""
PROMPT_DeepSeek = PromptTemplate.from_template(template_DeepSeek)

# Initialize Gemini model with token limit
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    # max_output_tokens=100  # <-- Limits response to ~50 words
)
memory_DeepSeek = ConversationBufferMemory()
conversation_DeepSeek = ConversationChain(
    llm=llm,
    memory=memory_DeepSeek,
    prompt=PROMPT_DeepSeek  # <-- Using custom prompt
)


@app.route('/deepseek', methods=['GET', 'POST'])
def deepseek():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        selected_model = request.form.get('selected_model', 'Gemini')
        
        if selected_model == 'Gemini':
            try:
                response = conversation_DeepSeek.predict(input=user_input)
                # Convert Markdown to HTML
                html_response = markdown.markdown(response, extensions=['fenced_code'])
            except Exception as e:
                html_response = f"<div class='error'>Error: {str(e)}</div>"
        else:
            html_response = "This model is not yet implemented"

        session['chat_history'].append({'type': 'human', 'content': user_input})
        session['chat_history'].append({'type': 'ai', 'content': html_response})
        session.modified = True


    

    return render_template('deepseek.html', chat_history=session['chat_history'])




@app.route('/reset', methods=['POST'])
def reset_chat():
    conversation.memory.clear()
    session['chat_history'] = []
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
