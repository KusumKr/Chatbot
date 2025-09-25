from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import openai
import os
import logging
from datetime import datetime
import time
import random
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAX_MESSAGE_LENGTH = 1000
    MAX_CONVERSATION_HISTORY = 10
    SUPPORTED_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    DEFAULT_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 128

# In-memory conversation storage (for demo - use Redis/DB in production)
conversations: Dict[str, List[Dict]] = {}

class ChatbotService:
    """Service class to handle chatbot logic"""

    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    def get_response(self, message: str, conversation_id: str = "default",
                    model: str = Config.DEFAULT_MODEL) -> Dict:
        try:
            if len(message.strip()) == 0:
                return {"error": "Message cannot be empty"}

            if len(message) > Config.MAX_MESSAGE_LENGTH:
                return {"error": f"Message too long. Maximum {Config.MAX_MESSAGE_LENGTH} characters allowed"}

            if conversation_id not in conversations:
                conversations[conversation_id] = []

            conversation_history = conversations[conversation_id]

            messages = [
                {"role": "system", "content": "You are a helpful, knowledgeable assistant. Provide clear, concise, and accurate responses."}
            ]

            recent_history = conversation_history[-Config.MAX_CONVERSATION_HISTORY:]
            for msg in recent_history:
                messages.extend([
                    {"role": "user", "content": msg["user"]},
                    {"role": "assistant", "content": msg["bot"]}
                ])

            messages.append({"role": "user", "content": message})

            # Call OpenAI with retries to handle transient rate limits
            last_error = None
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=Config.MAX_TOKENS,
                        temperature=0.7,
                        top_p=0.9
                    )
                    break
                except openai.RateLimitError as e:
                    last_error = e
                    # Exponential backoff with jitter: ~0.8-1.2s, 1.6-2.4s, ...
                    wait = (0.8 + random.random() * 0.4) * (2 ** attempt)
                    logger.warning(f"OpenAI rate limited, retrying in {wait:.2f}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
            else:
                logger.warning("OpenAI rate limit exceeded after retries")
                return {"error": "Rate limit exceeded. Please wait a few seconds and try again."}

            bot_response = response.choices[0].message.content.strip()

            conversation_history.append({
                "user": message,
                "bot": bot_response,
                "timestamp": datetime.now().isoformat(),
                "model": model
            })

            logger.info(f"Conversation {conversation_id}: User message processed successfully")

            return {
                "reply": bot_response,
                "model_used": model,
                "timestamp": datetime.now().isoformat(),
                "conversation_length": len(conversation_history)
            }

        except openai.RateLimitError:
            logger.warning("OpenAI rate limit exceeded")
            return {"error": "Rate limit exceeded. Please try again later."}

        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed")
            return {"error": "Authentication failed. Please check API key."}

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return {"error": "An unexpected error occurred. Please try again."}

# Initialize chatbot service
try:
    chatbot_service = ChatbotService()
    logger.info("Chatbot service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot service: {str(e)}")
    chatbot_service = None

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service_available": chatbot_service is not None
    })

@app.route("/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field in request"}), 400

        if not chatbot_service:
            return jsonify({"error": "Chatbot service unavailable"}), 503

        user_message = data["message"].strip()
        conversation_id = data.get("conversation_id", "default")
        model = data.get("model", Config.DEFAULT_MODEL)

        if model not in Config.SUPPORTED_MODELS:
            model = Config.DEFAULT_MODEL

        result = chatbot_service.get_response(user_message, conversation_id, model)

        if "error" in result:
            # Map known errors to appropriate HTTP status codes
            error_msg = str(result.get("error", "")).lower()
            if "rate limit" in error_msg:
                return jsonify(result), 429
            if "authentication" in error_msg:
                return jsonify(result), 401
            if "unavailable" in error_msg:
                return jsonify(result), 503
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        return jsonify({"conversation": [], "total_messages": 0})

    conversation = conversations[conversation_id]
    return jsonify({
        "conversation": conversation,
        "total_messages": len(conversation),
        "conversation_id": conversation_id
    })

@app.route("/conversation/<conversation_id>", methods=["DELETE"])
def clear_conversation(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
        logger.info(f"Conversation {conversation_id} cleared")

    return jsonify({"message": "Conversation cleared successfully"})

@app.route("/stats")
def get_stats():
    total_conversations = len(conversations)
    total_messages = sum(len(conv) for conv in conversations.values())

    return jsonify({
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "active_conversations": list(conversations.keys()),
        "supported_models": Config.SUPPORTED_MODELS
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please slow down."
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end."
    }), 500

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
        threaded=True
    )
