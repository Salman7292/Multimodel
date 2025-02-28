from flask import Flask, render_template, request, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
import markdown  # Add this import

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-123")

# Configure Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDkrNYC_6LiCrsvNKlsAe9KR6tG-CMqr0M"

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
    temperature=0.7,
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

    return render_template('2.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)