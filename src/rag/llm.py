import logging
import os

import requests

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def _fallback_answer(question, matches):
    lines = "\n".join(f"- {m['image']}: {m['caption']}" for m in matches)
    return f'Photos matching "{question}":\n{lines}'


def generate_answer(question, matches):
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        logger.info("GROQ_API_KEY not set — returning retrieved captions without LLM generation.")
        return _fallback_answer(question, matches)

    context = "\n".join(f"- {m['image']}: {m['caption']}" for m in matches)
    prompt = (
        "You are a helpful assistant answering questions about a photo "
        "collection. Use only the captions below, which are automatically "
        "generated descriptions of specific photos, to answer the "
        "question. Mention which image filenames are relevant.\n\n"
        f"Captions:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        logger.exception("LLM call failed — falling back to retrieved captions.")
        return _fallback_answer(question, matches)
