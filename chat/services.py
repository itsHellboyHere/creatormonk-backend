import os
import json
import boto3
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

def get_secrets():
    if os.getenv("ENV") == "local":
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "DEBUG": os.getenv("DEBUG", "True")
        }
    client = boto3.client("secretsmanager", region_name="ap-south-1")
    secret = client.get_secret_value(SecretId="creatormonk/backend")
    return json.loads(secret["SecretString"])

secrets = get_secrets()
HF_TOKEN = secrets["HF_TOKEN"] 
# print(f"HF_TOKEN loaded: {HF_TOKEN[:10]}...") 

from huggingface_hub import login
login(token=HF_TOKEN)


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
# ── KNOWLEDGE BASE ────────────────────────────────────────────────
# This is CreatorMonk's real information
#  expand this with your actual services, pricing, process
documents = [

# ───────── BRAND IDENTITY ─────────
"CreatorMonk is a digital growth agency founded in 2026.",
"CreatorMonk helps creators, founders and personal brands become digital superstars.",
"CreatorMonk focuses on long term brand authority instead of short term algorithm hacks.",
"The Monk Way is CreatorMonk’s philosophy for sustainable online growth.",
"Mission Superstar is CreatorMonk’s internal framework for scaling creators.",
"CreatorMonk operates through strategy, technology and content systems.",
"CreatorMonk follows a 3 phase system: Neev, Asli Power and Dhamaka.",

# ───────── SOCIAL_OS OVERVIEW ─────────
"Social_OS is CreatorMonk’s Instagram and social media growth system.",
"Social_OS focuses on Instagram growth and Reels growth.",
"CreatorMonk provides cross platform social media scaling.",
"CreatorMonk manages Instagram accounts professionally.",
"CreatorMonk manages LinkedIn accounts for personal branding.",
"CreatorMonk manages Twitter accounts for audience growth.",
"CreatorMonk creates structured content calendars for social media.",
"CreatorMonk writes captions optimized for engagement.",
"CreatorMonk designs graphics for Instagram posts and reels.",
"CreatorMonk provides monthly social media analytics reports.",

# ───────── INSTA LAUNCH PLAN ─────────
"Insta Launch plan costs ₹14,999.",
"Insta Launch includes Instagram profile optimization.",
"Insta Launch includes reels content strategy.",
"Insta Launch includes hashtag research.",
"Insta Launch includes posting schedule setup.",
"Insta Launch is designed for small creators starting growth.",

# ───────── REELS GROWTH PLAN ─────────
"Reels Growth plan costs ₹24,999.",
"Reels Growth includes viral reels strategy.",
"Reels Growth includes reels editing optimization.",
"Reels Growth includes trend and audio research.",
"Reels Growth includes audience engagement tactics.",
"Reels Growth focuses on scaling Instagram reach quickly.",

# ───────── SOCIAL SCALE PLAN ─────────
"Social Scale plan costs ₹49,999.",
"Social Scale includes cross platform promotion.",
"Social Scale includes monetization strategy.",
"Social Scale includes brand collaboration setup.",
"Social Scale includes one to one growth mentorship.",
"Social Scale is designed for serious creators scaling fast.",

# ───────── FILM_OS OVERVIEW ─────────
"Film_OS is CreatorMonk’s YouTube growth and video production system.",
"CreatorMonk provides professional YouTube video editing.",
"CreatorMonk edits Instagram Reels and short form content.",
"Video turnaround time is 48 to 72 hours.",
"CreatorMonk optimizes YouTube thumbnails for higher CTR.",
"CreatorMonk performs full YouTube channel audits.",
"CreatorMonk builds custom YouTube content strategies.",

# ───────── YOUTUBE STARTER PLAN ─────────
"YouTube Starter plan costs ₹10,000 per month.",
"YouTube Starter includes 4 long form videos per month.",
"YouTube Starter includes SEO optimized titles.",
"YouTube Starter includes optimized descriptions.",
"YouTube Starter includes basic thumbnail design.",

# ───────── YOUTUBE GROWTH PLAN ─────────
"YouTube Growth plan costs ₹20,000 per month.",
"YouTube Growth includes 8 long form videos per month.",
"YouTube Growth includes advanced SEO optimization.",
"YouTube Growth includes thumbnail optimization for CTR.",
"YouTube Growth includes analytics performance reports.",

# ───────── SOFT_OS OVERVIEW ─────────
"Soft_OS is CreatorMonk’s software and website development system.",
"CreatorMonk builds websites using Next.js and Django.",
"CreatorMonk develops full stack web applications.",
"CreatorMonk builds mobile responsive websites.",
"CreatorMonk builds SEO optimized websites.",
"CreatorMonk handles frontend and backend development.",
"CreatorMonk designs scalable backend APIs.",
"CreatorMonk builds real time chat systems.",
"CreatorMonk configures databases and object storage.",
"CreatorMonk deploys projects securely to production.",

# ───────── STARTER DEVELOPMENT PLAN ─────────
"Creator Monk Starter plan costs ₹10,000.",
"Starter plan includes Next.js frontend development.",
"Starter plan includes basic backend service.",
"Starter plan includes database configuration.",
"Starter plan includes project deployment.",

# ───────── GROWTH DEVELOPMENT PLAN ─────────
"Creator Monk Growth plan costs ₹25,000.",
"Growth plan includes advanced backend API architecture.",
"Growth plan includes search container integration.",
"Growth plan includes database optimization.",
"Growth plan includes secure production deployment.",

# ───────── PLATFORM DEVELOPMENT PLAN ─────────
"Creator Monk Platform plan costs ₹30,000.",
"Platform plan includes scalable search system.",
"Platform plan includes full backend infrastructure.",
"Platform plan includes database and object storage.",
"Platform plan includes real time chat service.",
"Platform plan includes production ready architecture.",

# ───────── ADS & PERFORMANCE MARKETING ─────────
"CreatorMonk runs Facebook Ads campaigns.",
"CreatorMonk runs Google Ads campaigns.",
"CreatorMonk reduces cost per lead through optimization.",
"CreatorMonk focuses on maximizing ROI from ads.",
"Ad management includes campaign setup.",
"Ad management includes audience targeting.",
"Ad management includes A/B testing creatives.",
"Ad management includes weekly performance reporting.",
"CreatorMonk helps ecommerce brands scale paid ads.",

# ───────── PROCESS & PRICING ─────────
"CreatorMonk offers one time project pricing.",
"CreatorMonk offers monthly retainer packages.",
"Monthly retainers include a dedicated account manager.",
"Monthly retainers include weekly update calls.",
"Onboarding process takes 3 to 5 days.",
"Onboarding starts with a discovery call.",
"CreatorMonk builds custom strategies for each client.",
"Website pricing depends on project complexity.",

# ───────── CREDIBILITY ─────────
"CreatorMonk has worked with over 50 creators and brands.",
"CreatorMonk has case studies in YouTube growth.",
"CreatorMonk has case studies in ecommerce website development.",
"CreatorMonk has case studies in paid ad scaling.",
"CreatorMonk focuses on sustainable brand building.",

# ───────── FAQ STYLE VARIATIONS ─────────
"What is the price of Instagram growth services?",
"Instagram growth plans start at ₹14,999.",
"What is the cost of YouTube management?",
"YouTube management starts at ₹10,000 per month.",
"Does CreatorMonk build websites?",
"CreatorMonk builds full stack websites.",
"Does CreatorMonk manage paid ads?",
"CreatorMonk manages Facebook and Google Ads.",
"How long does onboarding take?",
"Onboarding takes approximately 3 to 5 days.",
"What platforms does CreatorMonk manage?",
"CreatorMonk manages Instagram, YouTube, LinkedIn and Twitter.",
"Does CreatorMonk provide backend development?",
"CreatorMonk provides backend APIs and real time systems.",
]

# ── LOAD MODELS ONCE ON SERVER START ─────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)
print("RAG system ready!")


def retrieve(query, top_k=3, threshold=0.3):
    query_embedding = embedder.encode(query)
    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

    paired = []
    for idx, score in enumerate(similarities):
        paired.append((idx, score))

    paired.sort(key=lambda x: x[1], reverse=True)
    top_results = paired[:top_k]

    relevant = []
    for idx, score in top_results:
        if score > threshold:
            relevant.append(documents[idx])

    return relevant


def generate_answer(question, context_docs):
    context = "\n".join([f"- {doc}" for doc in context_docs])

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for CreatorMonk, a creator agency. Answer using ONLY the context provided. If context is insufficient say 'I don't have that information, please contact us directly at creatormonk.in'. Keep answers concise and friendly."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def chat(question):
    relevant_docs = retrieve(question)

    if not relevant_docs:
        return "I don't have information about that. Please contact us directly at creatormonk.in!"

    answer = generate_answer(question, relevant_docs)
    return answer