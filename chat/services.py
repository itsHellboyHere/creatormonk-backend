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
MAX_HISTORY_TURNS = 6
MAX_TOKENS = 120
MAX_SENTENCES = 3

_client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
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
    "lagega", "shuru", "paisa", "paise", "matlab", "thik", "theek", "insaan",
}


def detect_language(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hindi"
    if set(re.findall(r"[a-z]+", text.lower())) & HINGLISH_HINTS:
        return "hinglish"
    return "english"


def _wa():
    return settings.WHATSAPP_NUMBER.removeprefix("91")


# ---------------------------------------------------------------- intents
# Some messages have nothing to retrieve. Answering them from the knowledge
# base produces the wrong reply every time, so handle them directly.

GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|namaste|namaskar|salaam|हाय|हैलो|नमस्ते)[\s!.,]*$",
    re.I,
)

GOODBYE_RE = re.compile(
    r"^\s*(bye+|goodbye|good bye|see ya|see you|tata|ta ta|alvida|"
    r"ok bye|okay bye|thanks bye|thank you bye|"
    r"अलविदा|बाय)[\s!.,]*$",
    re.I,
)

THANKS_RE = re.compile(
    r"^\s*(thanks|thank you|thx|ty|shukriya|dhanyavad|dhanyawad|"
    r"धन्यवाद|शुक्रिया)[\s!.,]*$",
    re.I,
)

HUMAN_RE = re.compile(
    r"(talk|speak|connect|chat)\s+(to|with)\s+(a\s+)?(human|person|someone|"
    r"real\s+person|agent|team)|human\s+(agent|support)|"
    r"real\s+person|insaan\s+se|kisi\s+se\s+baat|team\s+se\s+baat|"
    r"इंसान|असली\s+व्यक्ति",
    re.I,
)


CANNED = {
    "greeting": {
        "english": "Hey! What are you looking to build?",
        "hinglish": "Hey! Kya banwana chahte ho?",
        "hindi": "नमस्ते! आप क्या बनवाना चाहते हैं?",
    },
    "goodbye": {
        "english": "Thanks for stopping by! WhatsApp us at +91 {wa} whenever you're ready.",
        "hinglish": "Milke accha laga! Jab bhi ready ho, +91 {wa} par WhatsApp kar dena.",
        "hindi": "मिलकर अच्छा लगा! जब भी तैयार हों, +91 {wa} पर WhatsApp कर दें।",
    },
    "thanks": {
        "english": "Anytime! Anything else you'd like to know?",
        "hinglish": "Anytime! Aur kuch jaanna hai?",
        "hindi": "कभी भी! और कुछ जानना है?",
    },
    "human": {
        "english": "Of course — WhatsApp is fastest: +91 {wa}. You can also email {email}.",
        "hinglish": "Bilkul — WhatsApp sabse fast hai: +91 {wa}. Ya {email} par mail kar do.",
        "hindi": "बिलकुल — WhatsApp सबसे तेज़ है: +91 {wa}। या {email} पर मेल कर दें।",
    },
}


def detect_intent(text: str):
    t = text.strip()
    if HUMAN_RE.search(t):
        return "human"
    if GREETING_RE.match(t):
        return "greeting"
    if GOODBYE_RE.match(t):
        return "goodbye"
    if THANKS_RE.match(t):
        return "thanks"
    return None


# ---------------------------------------------------------------- contact

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE_RE = re.compile(r"(?:\+?91[\s-]?|0)?[6-9]\d{9}\b")


def extract_contact(text: str):
    cleaned = re.sub(r"[\s()-]", "", text or "")
    email = EMAIL_RE.search(text or "")
    phone = PHONE_RE.search(cleaned)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
    }


def contact_from_history(history, question):
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
    "english": "I don't have that one — WhatsApp us at +91 {wa} or email {email} and the team will reply.",
    "hinglish": "Ye baat mere paas nahi hai — +91 {wa} par WhatsApp karo ya {email} par mail bhejo.",
    "hindi": "यह जानकारी मेरे पास नहीं है — +91 {wa} पर WhatsApp करें या {email} पर ईमेल करें।",
}

BUSY = {
    "english": "I'm having trouble right now. Please WhatsApp us at +91 {wa} or email {email}.",
    "hinglish": "Abhi thodi dikkat aa rahi hai. +91 {wa} par WhatsApp karo ya {email} par mail karo.",
    "hindi": "अभी कुछ दिक्कत आ रही है। +91 {wa} पर WhatsApp करें या {email} पर ईमेल करें।",
}


def _fill(t):
    return t.format(wa=_wa(), email=settings.CONTACT_EMAIL)


# ---------------------------------------------------------------- retrieval

FOLLOW_UPS = {
    "yes", "yeah", "yep", "ok", "okay", "sure", "haan", "han", "ha", "hmm",
    "thik", "theek", "achha", "acha", "sahi", "done", "please", "plz",
    "more", "batao", "aur batao",
}


def build_query(question: str, history) -> str:
    q = question.strip()
    if len(q.split()) > 3 and q.lower() not in FOLLOW_UPS:
        return q
    for turn in reversed(history):
        if turn.get("role") == "user":
            prev = (turn.get("text") or "").strip()
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


# ---------------------------------------------------------------- trimming

SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def trim(text: str, limit: int = MAX_SENTENCES) -> str:
    """The prompt asks for short answers; this guarantees them."""
    text = (text or "").strip()
    parts = [p for p in SENT_SPLIT.split(text) if p.strip()]
    if len(parts) <= limit:
        return text
    return " ".join(parts[:limit]).strip()


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You are the assistant on CreatorMonk's website — a web, AI and automation agency in Greater Noida, India.

LENGTH — the single most important rule:
Maximum 2 sentences. Never 3. Never a paragraph, never bullet points. Ask at most ONE short question, and only when it genuinely moves things forward. If you can answer in one sentence, do.

ANSWER THE ACTUAL MESSAGE:
Respond to what the visitor just said, not to something earlier in the conversation. Never repeat a reply you have already given.

FACTS:
Use only the CONTEXT. Never invent services, platforms, tools, timelines, guarantees or claims. If something is not in the CONTEXT, do not mention it. If the CONTEXT doesn't cover the question, say you don't have that detail and point to WhatsApp or email.

WHAT YOU CANNOT DO:
You cannot book calls, check calendars, send emails or access any system. Never say "I'll schedule", "let me arrange", or "I'll have someone call you at a time that suits you". Only say the team will reach out.

PRICING:
Never give a price, range, estimate or comparison — not even "depends on size" or "starts from". Say every project is different and a clear quote comes after a short call.

LANGUAGE — match the visitor exactly:
Devanagari Hindi -> Devanagari Hindi. Hinglish -> Hinglish, casual and natural. English -> English. Never mix scripts in one reply.

TONE:
Warm, plain, honest. Never salesy."""


def generate_answer(question, docs, lang, history, contact, contact_is_new):
    context = "\n".join(f"- {d}" for d in docs)

    lang_note = {
        "hindi": "Visitor wrote in Devanagari Hindi. Reply in Devanagari Hindi.",
        "hinglish": "Visitor wrote in Hinglish. Reply in Hinglish.",
        "english": "Visitor wrote in English. Reply in English.",
    }[lang]

    if contact_is_new:
        contact_note = (
            "The visitor just shared their contact details in THIS message. "
            "Thank them in one short sentence and say the team will reach out. Nothing more."
        )
    elif contact["email"] or contact["phone"]:
        contact_note = (
            "The visitor already shared contact details earlier in this conversation. "
            "Do NOT ask for an email or phone number again, and do NOT thank them for it "
            "again — that was already done. Just answer their current message normally."
        )
    else:
        contact_note = (
            "No contact details shared yet. If — and only if — the visitor seems ready to "
            "start, you may ask once for an email or phone number."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
            "Reply in 2 sentences maximum. Answer this message specifically."
        ),
    })

    res = _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        presence_penalty=0.4,
        frequency_penalty=0.4,
    )
    return trim(res.choices[0].message.content)


# ---------------------------------------------------------------- public API

def chat(question: str, history=None) -> dict:
    question = (question or "").strip()
    history = history or []
    lang = detect_language(question)

    if not question:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    contact_now = extract_contact(question)
    contact_is_new = bool(contact_now["email"] or contact_now["phone"])
    contact = contact_from_history(history, question)

    # Fixed replies for messages the knowledge base can't help with.
    # Skipped when the visitor also handed over a phone/email in the same line.
    if not contact_is_new:
        intent = detect_intent(question)
        if intent:
            logger.info("intent=%s lang=%s", intent, lang)
            return {
                "answer": _fill(CANNED[intent][lang]),
                "language": lang,
                "grounded": True,
                "intent": intent,
                "contact": contact,
            }

    try:
        docs = retrieve(build_query(question, history))
    except Exception:
        logger.exception("Retrieval failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    if contact_is_new and not docs:
        thanks = {
            "english": "Got it — thanks! The team will reach out shortly.",
            "hinglish": "Mil gaya, thanks! Team jaldi contact karegi.",
            "hindi": "मिल गया, धन्यवाद! टीम जल्द संपर्क करेगी।",
        }[lang]
        return {"answer": thanks, "language": lang, "grounded": True, "contact": contact}

    if not docs:
        return {"answer": _fill(FALLBACKS[lang]), "language": lang, "grounded": False}

    try:
        answer = generate_answer(question, docs, lang, history, contact, contact_is_new)
    except Exception:
        logger.exception("LLM call failed")
        return {"answer": _fill(BUSY[lang]), "language": lang, "grounded": False}

    return {"answer": answer, "language": lang, "grounded": True, "contact": contact}