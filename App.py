from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from groq import Groq
import google.generativeai as genai
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# FIX 1: Require a stable SECRET_KEY — os.urandom() changes on every restart
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable must be set")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

groq_key = os.environ.get("GROQ_API_KEY", "")
gemini_key = os.environ.get("GEMINI_API_KEY", "")
app_password = os.environ.get("APP_PASSWORD", "")

# FIX 5: Validate API keys at startup
if not groq_key:
    raise RuntimeError("GROQ_API_KEY environment variable must be set")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY environment variable must be set")

genai.configure(api_key=gemini_key)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if app_password and pwd == app_password:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Incorrect password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def ask():
    try:
        # FIX 3: Check that JSON body is present
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        # FIX 2: Reject empty prompts
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        history = data.get("history", [])[-20:]

        groq_answer = ask_groq(prompt, history)
        gemini_answer = ask_gemini(prompt, history)
        best_answer = pick_best(prompt, groq_answer, gemini_answer)

        # FIX 4: Renamed "gpt" key to "groq" (it's Groq/LLaMA, not GPT)
        return jsonify({
            "groq": groq_answer,
            "gemini": gemini_answer,
            "best": best_answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def ask_groq(prompt, history):
    client = Groq(api_key=groq_key)
    messages = history + [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1200
    )
    return response.choices[0].message.content

def ask_gemini(prompt, history):
    gem_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gem_history.append({"role": role, "parts": [msg["content"]]})

    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=gem_history)
    response = chat.send_message(prompt)
    return response.text

def pick_best(prompt, groq_answer, gemini_answer):
    try:
        client = Groq(api_key=groq_key)

        judge = f"""
User Prompt:
{prompt}

Answer A:
{groq_answer}

Answer B:
{gemini_answer}

Return only the best final answer.
"""

        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": judge}],
            temperature=0.3,
            max_tokens=1200
        )

        return result.choices[0].message.content
    except Exception:
        return groq_answer

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "groq": bool(groq_key),
        "gemini": bool(gemini_key)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
