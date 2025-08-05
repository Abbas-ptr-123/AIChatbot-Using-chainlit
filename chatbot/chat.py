import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import json
import requests
from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GEMINI_API_KEY or not NEWS_API_KEY:
    raise ValueError("GEMINI_API_KEY or NEWS_API_KEY is not set.")

# Set up Gemini client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Function to fetch real news from NewsAPI
def get_real_news(category):
    if category == "cryptocurrency":
        # Use /v2/everything for cryptocurrency as it's not a standard category
        url = f"https://newsapi.org/v2/everything?q={category}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt"
    else:
        # Use /v2/top-headlines for standard categories
        url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWS_API_KEY}&language=en"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to fetch news: HTTP {response.status_code} - {response.json().get('message', 'Unknown error')}"
        articles = response.json().get("articles", [])[:10]
        return "\n".join([f"- {article['title']}" for article in articles if 'title' in article])
    except Exception as e:
        return f"Failed to fetch news: {e}"

# Helper to summarize conversation history
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
    result = Runner.run_sync(
        starting_agent=Agent(name="Summarizer", instructions="Summarize conversation history", model=model),
        input=prompt,
        run_config=config
    )
    return result.final_output

# Helper to call LLM to summarize news
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
    result = Runner.run_sync(
        starting_agent=Agent(name="Summarizer", instructions="Summarize news", model=model),
        input=prompt,
        run_config=config
    )
    return result.final_output

# On chat start
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
    cl.user_session.set("config", config)
    cl.user_session.set("agent", Agent(name="Triage Agent", instructions="You are a news triage assistant", model=model))
    await cl.Message(content="👋 Hi! Ask me about tech, health, business, or crypto news, or tell me something about yourself!").send()

# Main message handler
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

    result = Runner.run_sync(
        starting_agent=Agent(name="Triage Agent", instructions="Classify message with history context", model=model),
        input=classification_prompt,
        run_config=config
    )

    topic = result.final_output.strip().lower()

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

        result = Runner.run_sync(
            starting_agent=Agent(name="General Assistant", instructions="Answer general queries with context from chat history", model=model),
            input=general_prompt,
            run_config=config
        )
        response = result.final_output

    # Append the assistant's response to the history
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("chat_history", history)

    await cl.Message(content=response).send()

# On chat end: Save chat to file
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