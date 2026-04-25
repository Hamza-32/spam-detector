import json
import os
import re
import urllib.error
import urllib.request

from .config import DEFAULT_OLLAMA_MODEL, OLLAMA_ENDPOINT, OPENAI_MODEL


def parse_llm_label(raw_text: str):
    words = re.sub(r"[^A-Z ]", " ", str(raw_text).upper()).split()
    for word in words:
        if word == "SPAM":
            return 1
        if word == "HAM":
            return 0
    return None


def predict_with_ollama(texts, model: str = DEFAULT_OLLAMA_MODEL):
    preds = []
    for text in texts:
        prompt = (
            "Classify this SMS as SPAM or HAM. Reply with exactly one word: SPAM or HAM.\\n\\n"
            f"SMS: {text}"
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            OLLAMA_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = parse_llm_label(obj.get("response", ""))
        preds.append(0 if parsed is None else parsed)
    return preds


def predict_with_openai(texts):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    preds = []

    for text in texts:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Classify an SMS as SPAM or HAM. Reply with exactly one token: SPAM or HAM.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
        )
        parsed = parse_llm_label(response.output_text or "")
        preds.append(0 if parsed is None else parsed)

    return preds
