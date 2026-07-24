import re
import logging
from functools import lru_cache

from django.conf import settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from .kb import DOCUMENTS

logger = logging.getLogger(__name__)

# Multilingual — the old all-MiniLM-L6-v2 was English-only, which is why
# Hindi and Devanagari questions never retrieved anything useful.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 6
THRESHOLD = 0.30

_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.HF_TOKEN,
    timeout=25.0,
    max_retries=1,
)


@lru_cache(maxsize=1)
def _engine():
    """Load the embedding model once, lazily, and cache the doc vectors."""
    logger.info("Loading embedding model %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(DOCUMENTS, normalize_embeddings=True, show_progress_bar=False)
    logger.info("Knowledge base ready — %d documents", len(DOCUMENTS))
    return model, vectors


# ---------------------------------------------------------------- language

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

HINGLISH_HINTS = {
    "kya", "kaise", "kitna", "kitne", "hai", "hain", "ho", "kar", "karo", "karte",
    "karta", "karna", "chahiye", "sakte", "sakta", "mujhe", "aap", "aapka", "apna",
    "nahi", "haan", "acha", "accha", "bhai", "batao", "bata", "kaam", "paisa",
    "kitna", "milega", "banate", "banata", "banao", "chalta", "hoga", "wala",
}


def detect_language(text: str) -> str:
    """Returns 'hindi', 'hinglish' or 'english'."""
    if DEVANAGARI.search(text):
        return "hindi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & HINGLISH_HINTS:
        return "hinglish"
    return "english"


# ---------------------------------------------------------------- fallbacks

FALLBACKS = {
    "english": (
        "I don't have that one on hand — but the team does. "
        "Message us on WhatsApp at +91 {wa} or email {email}, "
        "and you'll hear back within a few hours."
    ),
    "hinglish": (
        "Ye baat abhi mere paas nahi hai — par team ke paas hai. "
        "+91 {wa} par WhatsApp karo ya {email} par mail bhejo, "
        "kuch hi ghanton mein reply mil jayega."
    ),
    "hindi": (
        "यह जानकारी अभी मेरे पास नहीं है — लेकिन टीम के पास है। "
        "+91 {wa} पर WhatsApp करें या {email} पर ईमेल करें, "
        "कुछ ही घंटों में जवाब मिल जाएगा।"
    ),
}

BUSY = {
    "english": (
        "I'm having trouble thinking right now. Please message us on WhatsApp "
        "at +91 {wa} or email {email} — a real person will reply."
    ),
    "hinglish": (
        "Abhi mujhe sochne mein dikkat aa rahi hai. +91 {wa} par WhatsApp karo "
        "ya {email} par mail karo — ek asli insaan reply karega."
    ),
    "hindi": (
        "अभी मुझे जवाब देने में दिक्कत आ रही है। कृपया +91 {wa} पर WhatsApp करें "
        "या {email} पर ईमेल करें — एक असली व्यक्ति जवाब देगा।"
    ),
}


def _fill(template: str) -> str:
    return template.format(
        wa=settings.WHATSAPP_NUMBER.removeprefix("91"),
        email=settings.CONTACT_EMAIL,
    )


# ---------------------------------------------------------------- retrieval

def retrieve(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD):
    model, vectors = _engine()
    q = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(q, vectors)[0]

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    hits = [(DOCUMENTS[i], float(s)) for i, s in ranked if float(s) > threshold]

    logger.info("Query=%r top_score=%.3f hits=%d", query[:60],
                float(ranked[0][1]) if ranked else 0.0, len(hits))
    return [doc for doc, _ in hits]


# ---------------------------------------------------------------- generation

SYSTEM_PROMPT = """You are the assistant on CreatorMonk's website. CreatorMonk is a web, AI and automation agency in Greater Noida, India.

WHO YOU'RE TALKING TO
Someone visiting the site who is thinking about hiring CreatorMonk. Be warm, plain-spoken and short. Two or three sentences is usually enough. No corporate filler, no bullet lists unless they ask for a list.

LANGUAGE — match the visitor exactly:
- Hindi in Devanagari script -> reply fully in Devanagari Hindi
- Hinglish (Hindi written in English letters) -> reply in Hinglish, casual and friendly, the way a person actually types
- English -> reply in English
Never mix scripts in one reply.

FACTS
Use only the CONTEXT provided. Never invent services, timelines, client names or claims. If the context does not cover it, say you don't have that detail and point them to WhatsApp or email.

PRICING
Never state, estimate, hint at or compare any price, budget or range — not even "starts from" or "depends on size". Always say every project is different and one clear quote comes after a short call, then give the WhatsApp number and email.

WHEN THEY SOUND READY
If someone describes a project or asks how to start, tell them the next step is a short call, and ask if they'd like to leave an email or phone number so the team can reach out. Ask once, gently. Don't push.

TONE
Confident, honest, never salesy. It's fine to say CreatorMonk might not be the right fit."""


def generate_answer(question: str, context_docs, lang: str) -> str:
    context = "\n".join(f"- {doc}" for doc in context_docs)

    lang_note = {
        "hindi": "The visitor wrote in Hindi (Devanagari). Reply in Devanagari Hindi.",
        "hinglish": "The visitor wrote in Hinglish. Reply in Hinglish, casual and warm.",
        "english": "The visitor wrote in English. Reply in English.",
    }[lang]

    response = _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{lang_note}\n\nCONTEXT:\n{context}\n\nVISITOR: {question}",
            },
        ],
        max_tokens=320,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------- public API

def chat(question: str) -> dict:
    """Returns {'answer': str, 'language': str, 'grounded': bool}."""
    question = (question or "").strip()
    lang = detect_language(question)

    if not question:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    try:
        docs = retrieve(question)
    except Exception:
        logger.exception("Retrieval failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    if not docs:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    try:
        answer = generate_answer(question, docs, lang)
    except Exception:
        logger.exception("LLM call failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    return {"answer": answer, "language": lang, "grounded": True}