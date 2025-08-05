import os
from dotenv import load_dotenv
import chainlit as cl
import json
import requests
from datetime import datetime
import litellm

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Validate API keys
if not GEMINI_API_KEY or not NEWS_API_KEY:
    raise ValueError("GEMINI_API_KEY or NEWS_API_KEY is not set.")

# Configure LiteLLM for Gemini (Google AI Studio)
litellm.api_key = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # Explicitly set for Gemini provider

# Function to fetch news from NewsAPI
def get_real_news(category):
    # Map user-friendly category names to NewsAPI categories
    category_mapping = {
        "tech": "technology",
        "finance": "business",
        "health": "health"
    }
    newsapi_category = category_mapping.get(category, category)
    
    # Use /v2/top-headlines for all categories
    url = f"https://newsapi.org/v2/top-headlines?category={newsapi_category}&apiKey={NEWS_API_KEY}&language=en"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to fetch news: HTTP {response.status_code} - {response.json().get('message', 'Unknown error')}"
        articles = response.json().get("articles", [])[:10]
        return "\n".join([f"- {article['title']}" for article in articles if 'title' in article])
    except Exception as e:
        return f"Failed to fetch news: {e}"

# Helper to summarize conversation history using LiteLLM
def summarize_history(history):
    if not history:
        return ""
    prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the following conversation history concisely, focusing on key details like the user's name, preferences, or previous search topics."
            )
        },
        {"role": "user", "content": "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])}
    ]
    response = litellm.completion(
        model="gemini/gemini-2.0-flash",  # Explicitly use Gemini provider
        messages=prompt,
        temperature=0.7
    )
    return response.choices[0].message.content

# Helper to summarize news using LiteLLM
def summarize_with_llm(news_text, category, history):
    history_summary = summarize_history(history) if len(history) > 10 else ""
    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a smart news assistant. Summarize the following {category} news for a user. "
                "Use the conversation history summary to tailor the response if relevant."
                f"\nConversation history summary: {history_summary}" if history_summary else ""
            )
        },
        {"role": "user", "content": news_text}
    ] + history[-5:]  # Include last 5 messages for additional context
    response = litellm.completion(
        model="gemini/gemini-2.0-flash",  # Explicitly use Gemini provider
        messages=prompt,
        temperature=0.7
    )
    return response.choices[0].message.content

# Initialize chatbot on start
@cl.on_chat_start
async def start():
    # Initialize chat history
    history = []
    
    # Load previous history from file if it exists
    filename = "chat_history.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                all_history = json.load(f)
                # Get the most recent session's history
                if all_history:
                    history = all_history[-1]["session"]
            except json.JSONDecodeError:
                history = []

    cl.user_session.set("chat_history", history)
    await cl.Message(content="👋 Hi! Ask me about tech, health, business, or crypto news, or tell me something about yourself!").send()

# Handle incoming messages
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("chat_history", history)

    # Step 1: Classify the query with history context
    history_summary = summarize_history(history) if len(history) > 10 else ""
    classification_prompt = [
        {
            "role": "system",
            "content": (
                "Classify the user's message based on the conversation history and current input:\n"
                "- If the user wants the **latest news**, classify it as one of: 'news-tech', 'news-health', 'news-business', 'news-crypto'.\n"
                "- If the user is asking a **general question** (e.g., definition, explanation, or follow-up like 'What was my last search?'), return 'general'.\n"
                "Consider the conversation history to identify follow-up questions or context (e.g., user’s name or previous searches).\n"
                "Examples:\n"
                "- 'What is an ulcer?' → general\n"
                "- 'Tell me the latest about crypto' → news-crypto\n"
                "- 'Explain AI' → general\n"
                "- 'Any business updates?' → news-business\n"
                "- 'What was my last news search?' → general\n"
                f"\nConversation history summary: {history_summary}" if history_summary else ""
            )
        }
    ] + history[-5:] + [{"role": "user", "content": message.content}]

    classification_response = litellm.completion(
        model="gemini/gemini-2.0-flash",  # Explicitly use Gemini provider
        messages=classification_prompt,
        temperature=0.7
    )
    topic = classification_response.choices[0].message.content.strip().lower()

    valid_categories = {
        "news-tech": "technology",
        "news-health": "health",
        "news-business": "business",
        "news-crypto": "cryptocurrency"
    }

    # Step 2: If news category → fetch and summarize news
    if topic in valid_categories:
        category = valid_categories[topic]
        news = get_real_news(category)
        summary = summarize_with_llm(news, category, history)
        response = f"📰 Here is the latest **{category}** news:\n\n{summary}"

    # Step 3: If general question → answer with full history context
    else:
        history_summary = summarize_history(history) if len(history) > 10 else ""
        general_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based on the conversation history provided below. "
                    "If the user asks about information from previous messages (e.g., their name or past searches), use the history to respond accurately."
                    f"\nConversation history summary: {history_summary}" if history_summary else ""
                )
            }
        ] + history[-10:]  # Include last 10 messages for context

        general_response = litellm.completion(
            model="gemini/gemini-2.0-flash",  # Explicitly use Gemini provider
            messages=general_prompt,
            temperature=0.7
        )
        response = general_response.choices[0].message.content

    # Append the assistant's response to the history
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("chat_history", history)

    await cl.Message(content=response).send()

# Save history on chat end
@cl.on_chat_end
async def end():
    history = cl.user_session.get("chat_history")
    if not history:
        return

    filename = "chat_history.json"

    # Load old history if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                all_history = json.load(f)
            except json.JSONDecodeError:
                all_history = []
    else:
        all_history = []

    # Append this session's history
    all_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session": history
    })

    # Save the combined history back to the file
    with open(filename, "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"✅ Chat history saved to {filename}")