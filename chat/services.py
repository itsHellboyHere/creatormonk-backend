import re
import logging
from functools import lru_cache

from django.conf import settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from .kb import DOCUMENTS

logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
THRESHOLD = 0.32
MAX_HISTORY_TURNS = 6      # 3 exchanges — enough context, cheap tokens
MAX_TOKENS = 170           # short answers are the whole point

_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.HF_TOKEN,
    timeout=25.0,
    max_retries=1,
)


@lru_cache(maxsize=1)
def _engine():
    logger.info("Loading embedding model %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(DOCUMENTS, normalize_embeddings=True, show_progress_bar=False)
    logger.info("Knowledge base ready — %d documents", len(DOCUMENTS))
    return model, vectors


# ---------------------------------------------------------------- language

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

HINGLISH_HINTS = {
    "kya", "kaise", "kaisa", "kitna", "kitne", "hai", "hain", "ho", "kar", "karo",
    "karte", "karta", "karna", "chahiye", "sakte", "sakta", "mujhe", "aap", "aapka",
    "apna", "nahi", "haan", "acha", "accha", "bhai", "batao", "bata", "kaam",
    "milega", "banate", "banata", "banao", "chalta", "hoga", "wala", "lagta",
    "lagega", "shuru", "paisa", "paise", "matlab", "thik", "theek",
}


def detect_language(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hindi"
    if set(re.findall(r"[a-z]+", text.lower())) & HINGLISH_HINTS:
        return "hinglish"
    return "english"


# ---------------------------------------------------------------- contact detection

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
# Indian mobile: optional +91/0, then 10 digits starting 6-9
PHONE_RE = re.compile(r"(?:\+?91[\s-]?|0)?[6-9]\d{9}\b")


def extract_contact(text: str):
    """Returns {'email': str|None, 'phone': str|None} found in the text."""
    cleaned = re.sub(r"[\s()-]", "", text)
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(cleaned)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
    }


def contact_from_history(history, question):
    """Scan the whole conversation so we never ask twice."""
    found = {"email": None, "phone": None}
    for turn in list(history) + [{"role": "user", "text": question}]:
        if turn.get("role") != "user":
            continue
        got = extract_contact(turn.get("text", ""))
        found["email"] = found["email"] or got["email"]
        found["phone"] = found["phone"] or got["phone"]
    return found


# ---------------------------------------------------------------- fallbacks

FALLBACKS = {
    "english": "I don't have that one — message us on WhatsApp at +91 {wa} or email {email} and the team will reply.",
    "hinglish": "Ye baat mere paas nahi hai — +91 {wa} par WhatsApp karo ya {email} par mail bhejo, team reply kar degi.",
    "hindi": "यह जानकारी मेरे पास नहीं है — +91 {wa} पर WhatsApp करें या {email} पर ईमेल करें, टीम जवाब देगी।",
}

BUSY = {
    "english": "I'm having trouble right now. Please WhatsApp us at +91 {wa} or email {email}.",
    "hinglish": "Abhi thodi dikkat aa rahi hai. +91 {wa} par WhatsApp karo ya {email} par mail karo.",
    "hindi": "अभी कुछ दिक्कत आ रही है। +91 {wa} पर WhatsApp करें या {email} पर ईमेल करें।",
}


def _fill(t):
    return t.format(
        wa=settings.WHATSAPP_NUMBER.removeprefix("91"),
        email=settings.CONTACT_EMAIL,
    )


# ---------------------------------------------------------------- retrieval

FOLLOW_UPS = {
    "yes", "yeah", "yep", "ok", "okay", "sure", "haan", "han", "ha", "hmm",
    "thik", "theek", "achha", "acha", "sahi", "done", "please", "plz",
    "tell me more", "more", "aur batao", "batao",
}


def build_query(question: str, history) -> str:
    """A bare 'yeah' retrieves nothing useful. Anchor short replies to the
    last real question so retrieval still has something to work with."""
    q = question.strip()
    if len(q.split()) > 3 and q.lower() not in FOLLOW_UPS:
        return q

    for turn in reversed(history):
        if turn.get("role") == "user":
            prev = turn.get("text", "").strip()
            if len(prev.split()) > 3:
                return f"{prev} {q}"
    return q


def retrieve(query: str):
    model, vectors = _engine()
    q = model.encode(query, normalize_embeddings=True)
    scores = util.cos_sim(q, vectors)[0]

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]
    hits = [(DOCUMENTS[i], float(s)) for i, s in ranked if float(s) > THRESHOLD]

    logger.info("q=%r top=%.3f hits=%d", query[:60],
                float(ranked[0][1]) if ranked else 0.0, len(hits))
    return [d for d, _ in hits]


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You are the assistant on CreatorMonk's website — a web, AI and automation agency in Greater Noida, India.

LENGTH — this matters most:
Answer in 2 sentences. 3 only if truly needed. Never write a paragraph. Never use bullet points. Ask at most ONE question at the end, and only if it moves things forward. Short answers feel confident; long ones feel like a brochure.

FACTS:
Use only the CONTEXT below. Do not invent services, platforms, tools, timelines, guarantees or claims. If a platform, service or detail is not in the CONTEXT, do not mention it. If the CONTEXT doesn't cover the question, say you don't have that detail and point to WhatsApp or email.

WHAT YOU CANNOT DO:
You cannot book calls, check calendars, send emails or access any system. Never say "I'll schedule", "let me arrange" or "I'll have someone call you at a time that suits you". You can only say the team will reach out.

PRICING:
Never give a price, range, estimate or comparison — not even "depends on size" or "starts from". Say every project is different and a clear quote comes after a short call.

CONTACT DETAILS:
If the conversation already contains the visitor's email or phone number, thank them once, confirm the team will reach out, and NEVER ask for contact details again. Only ask for a contact detail if none has been given and the visitor is clearly interested.

LANGUAGE — match the visitor exactly:
Devanagari Hindi -> reply in Devanagari Hindi. Hinglish -> reply in Hinglish, casual and natural. English -> reply in English. Never mix scripts in one reply.

TONE:
Warm, plain, honest. Never salesy."""


def generate_answer(question, docs, lang, history, contact):
    context = "\n".join(f"- {d}" for d in docs)

    lang_note = {
        "hindi": "Visitor wrote in Devanagari Hindi. Reply in Devanagari Hindi.",
        "hinglish": "Visitor wrote in Hinglish. Reply in Hinglish.",
        "english": "Visitor wrote in English. Reply in English.",
    }[lang]

    if contact["email"] or contact["phone"]:
        given = " and ".join(
            p for p in [
                f"email ({contact['email']})" if contact["email"] else None,
                f"phone ({contact['phone']})" if contact["phone"] else None,
            ] if p
        )
        contact_note = (
            f"IMPORTANT: The visitor has ALREADY shared their {given}. "
            "Thank them briefly, confirm the team will reach out soon, and do NOT ask "
            "for an email or phone number again under any circumstances."
        )
    else:
        contact_note = "The visitor has not shared contact details yet."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # replay recent turns so the model actually remembers the conversation
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = "assistant" if turn.get("role") == "bot" else "user"
        text = (turn.get("text") or "").strip()
        if text:
            messages.append({"role": role, "content": text[:600]})

    messages.append({
        "role": "user",
        "content": (
            f"{lang_note}\n{contact_note}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"VISITOR: {question}\n\n"
            "Reply in 2 sentences maximum."
        ),
    })

    res = _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.4,
        presence_penalty=0.3,
    )
    return res.choices[0].message.content.strip()


# ---------------------------------------------------------------- public API

def chat(question: str, history=None) -> dict:
    question = (question or "").strip()
    history = history or []
    lang = detect_language(question)

    if not question:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    contact = contact_from_history(history, question)

    try:
        docs = retrieve(build_query(question, history))
    except Exception:
        logger.exception("Retrieval failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    # If they just handed over contact details, we don't need the KB at all —
    # and we must not let the model wander off into a fresh pitch.
    if (contact["email"] or contact["phone"]) and not docs:
        thanks = {
            "english": "Got it — thanks! The team will reach out to you shortly.",
            "hinglish": "Mil gaya, thanks! Team jaldi aapse contact karegi.",
            "hindi": "मिल गया, धन्यवाद! टीम जल्द ही आपसे संपर्क करेगी।",
        }[lang]
        return {
            "answer": thanks,
            "language": lang,
            "grounded": True,
            "contact": contact,
        }

    if not docs:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    try:
        answer = generate_answer(question, docs, lang, history, contact)
    except Exception:
        logger.exception("LLM call failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    return {
        "answer": answer,
        "language": lang,
        "grounded": True,
        "contact": contact,
    }