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

from huggingface_hub import login
login(token=HF_TOKEN)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ══════════════════════════════════════════════════════════════════
#   CREATORMONK KNOWLEDGE BASE 
#   NO PRICING in this document. Direct all pricing queries
#   to contact: +91 78273 32337 / hello@creatormonk.in
# ══════════════════════════════════════════════════════════════════
documents = [

    # ─────────────────────────────────────────
    # BRAND IDENTITY
    # ─────────────────────────────────────────
    "CreatorMonk is a full-stack digital agency founded in 2025.",
    "CreatorMonk is based in India and works with clients across the country.",
    "CreatorMonk helps creators, founders, startups and personal brands grow digitally.",
    "CreatorMonk operates across three core divisions: SOCIAL_OS, BUILD_OS and AUTO_OS.",
    "CreatorMonk focuses on long-term brand authority instead of short-term algorithm hacks.",
    "The Monk Way is CreatorMonk's philosophy for sustainable online growth.",
    "Mission Superstar is CreatorMonk's internal framework for scaling creators.",
    "CreatorMonk follows a 3-phase growth system: Neev (foundation), Asli Power (scale) and Dhamaka (dominance).",
    "CreatorMonk is a one-stop agency — clients do not need to hire multiple vendors.",
    "CreatorMonk has delivered work for over 30 clients and brands.",
    "CreatorMonk builds, grows and automates — all under one roof.",

    # ─────────────────────────────────────────
    # CONTACT INFORMATION
    # ─────────────────────────────────────────
    "To get pricing, contact CreatorMonk on WhatsApp at +91 78273 32337.",
    "To discuss a project, email CreatorMonk at hello@creatormonk.in.",
    "CreatorMonk responds to all inquiries within 2 hours.",
    "CreatorMonk working hours are Monday to Saturday, 10am to 8pm IST.",
    "You can book a free 30-minute strategy call with CreatorMonk.",
    "The free consultation call involves no sales pitch — just honest advice.",
    "To start a project with CreatorMonk, visit creatormonk.in/contact.",
    "For WhatsApp queries, message +91 78273 32337.",
    "For email queries, write to hello@creatormonk.in.",
    "Onboarding with CreatorMonk takes 3 to 5 business days after the initial call.",
    "CreatorMonk builds custom strategies for every client — no templates.",

    # ─────────────────────────────────────────
    # SOCIAL_OS — SOCIAL MEDIA & CONTENT
    # ─────────────────────────────────────────
    "SOCIAL_OS is CreatorMonk's social media and content division.",
    "SOCIAL_OS handles everything from scripting to posting to reporting.",
    "CreatorMonk manages Instagram accounts professionally.",
    "CreatorMonk creates and edits Reels and short-form videos.",
    "CreatorMonk writes scripts for Reels and YouTube Shorts.",
    "CreatorMonk manages YouTube channels including strategy and uploads.",
    "CreatorMonk manages LinkedIn accounts for personal branding.",
    "CreatorMonk manages Twitter and X accounts for audience growth.",
    "CreatorMonk manages Facebook pages and groups.",
    "CreatorMonk creates structured monthly content calendars.",
    "CreatorMonk writes captions and hooks optimized for engagement.",
    "CreatorMonk designs graphics and thumbnails for social media.",
    "CreatorMonk researches trending audios and formats every week.",
    "CreatorMonk provides monthly analytics and performance reports.",
    "CreatorMonk runs community management including DMs and comments.",
    "CreatorMonk runs paid ad campaigns on Meta — Facebook and Instagram.",
    "CreatorMonk runs paid ad campaigns on Google.",
    "CreatorMonk focuses on maximizing ROI and reducing cost per lead.",
    "CreatorMonk has delivered over 1 million organic reach for clients.",
    "SOCIAL_OS plans include Insta Launch, Reels Growth and Social Scale.",
    "For SOCIAL_OS pricing contact us at +91 78273 32337 or hello@creatormonk.in.",

    # ─────────────────────────────────────────
    # BUILD_OS — WEB, APPS & SOFTWARE
    # ─────────────────────────────────────────
    "BUILD_OS is CreatorMonk's web, app and software development division.",
    "CreatorMonk builds websites using Next.js for fast, modern frontends.",
    "CreatorMonk builds backend systems using Node.js and Python.",
    "CreatorMonk uses PostgreSQL and other databases for scalable data storage.",
    "CreatorMonk builds full-stack web applications from design to deployment.",
    "CreatorMonk builds SaaS platforms with authentication, billing and dashboards.",
    "CreatorMonk builds mobile apps for iOS and Android using React Native.",
    "CreatorMonk builds backend APIs that are scalable and secure.",
    "CreatorMonk builds e-commerce websites and custom shopping portals.",
    "CreatorMonk builds B2B client portals and internal tools.",
    "CreatorMonk deploys all projects to production with monitoring.",
    "CreatorMonk provides 30 days of post-launch support after every project.",
    "CreatorMonk designs UI and UX before writing a single line of code.",
    "CreatorMonk builds mobile-responsive, SEO-optimized websites.",
    "CreatorMonk integrates third-party APIs and payment gateways.",
    "CreatorMonk has shipped over 20 digital products.",
    "BUILD_OS plans include Starter Build, Growth Build and Platform Build.",
    "For BUILD_OS pricing contact us at +91 78273 32337 or hello@creatormonk.in.",

    # ─────────────────────────────────────────
    # AUTO_OS — AI & AUTOMATION
    # ─────────────────────────────────────────
    "AUTO_OS is CreatorMonk's AI and automation division.",
    "CreatorMonk builds custom AI agents powered by large language models.",
    "CreatorMonk builds RAG systems — Retrieval Augmented Generation — trained on company data.",
    "A RAG system allows a business to have an AI assistant that knows everything about their company.",
    "CreatorMonk built its own internal RAG-powered assistant running on its dashboard.",
    "CreatorMonk builds AI chatbots for customer support and sales.",
    "CreatorMonk automates business workflows using tools like n8n and custom pipelines.",
    "CreatorMonk eliminates repetitive manual work through intelligent automation.",
    "CreatorMonk integrates AI into existing tools like CRMs.",
    "CreatorMonk builds data pipelines and automated reporting systems.",
    "CreatorMonk connects third-party APIs so teams stop copy-pasting between tools.",
    "CreatorMonk builds full-stack AI-powered software with custom dashboards.",
    "CreatorMonk AI systems run 24 hours a day 7 days a week without manual input.",
    "AUTO_OS plans include Automation Starter, AI Agent Build and Full AI Stack.",
    "The Full AI Stack plan includes full-stack AI software with a custom dashboard.",
    "RAG systems built by CreatorMonk are trained on the client's own data and documents.",
    "For AUTO_OS pricing contact us at +91 78273 32337 or hello@creatormonk.in.",

    # ─────────────────────────────────────────
    # TECH STACK & INFRASTRUCTURE
    # ─────────────────────────────────────────
    "CreatorMonk uses Next.js for frontend web development.",
    "CreatorMonk uses Python and Node.js for backend development.",
    "CreatorMonk uses PostgreSQL as the primary relational database.",
    "CreatorMonk uses Docker for containerized deployments.",
    "CreatorMonk deploys backend services on AWS EC2.",
    "CreatorMonk uses GitHub Actions for CI/CD automated deployments.",
    "CreatorMonk uses AWS Secrets Manager for secure credential storage.",
    "CreatorMonk uses Cloudinary for media storage and video delivery.",
    "CreatorMonk uses Resend for transactional email delivery.",
    "CreatorMonk uses SentenceTransformers for AI embedding models.",
    "CreatorMonk uses Llama 3.1 8B Instruct via HuggingFace for AI responses.",
    "CreatorMonk's AI backend is deployed on EC2 with Docker Compose.",

    # ─────────────────────────────────────────
    # PROCESS
    # ─────────────────────────────────────────
    "CreatorMonk's process starts with a free discovery call.",
    "After the discovery call CreatorMonk builds a custom strategy.",
    "CreatorMonk presents designs and wireframes before development begins.",
    "CreatorMonk works in agile sprints with weekly client check-ins.",
    "CreatorMonk provides weekly progress updates on all active projects.",
    "CreatorMonk does a full brand audit before starting any social media work.",
    "CreatorMonk audits existing workflows before building any automation.",
    "CreatorMonk stress-tests all systems before launch.",
    "CreatorMonk provides documentation for everything it builds.",
    "Video editing turnaround time is 48 to 72 hours.",
    "Website projects are delivered with a full QA pass across devices and browsers.",

    # ─────────────────────────────────────────
    # FAQ STYLE — COMMON QUESTIONS
    # ─────────────────────────────────────────
    "What does CreatorMonk do?",
    "CreatorMonk is a full-stack digital agency offering social media management, web and software development, and AI automation.",
    "How much does CreatorMonk charge?",
    "For pricing please contact CreatorMonk at +91 78273 32337 or hello@creatormonk.in.",
    "What is the price of Instagram growth services?",
    "Please contact us at +91 78273 32337 or hello@creatormonk.in for current pricing.",
    "What is the cost of website development?",
    "Website pricing depends on scope. Contact us at +91 78273 32337 or hello@creatormonk.in.",
    "Does CreatorMonk build mobile apps?",
    "Yes, CreatorMonk builds iOS and Android apps using React Native.",
    "Does CreatorMonk manage paid ads?",
    "Yes, CreatorMonk manages Facebook, Instagram and Google Ads campaigns.",
    "How long does onboarding take?",
    "Onboarding takes approximately 3 to 5 business days after the initial discovery call.",
    "What platforms does CreatorMonk manage?",
    "CreatorMonk manages Instagram, YouTube, LinkedIn, Twitter and Facebook.",
    "Does CreatorMonk provide backend development?",
    "Yes, CreatorMonk builds backend APIs, databases and real-time systems.",
    "What is a RAG system?",
    "A RAG system is an AI assistant trained on your company's own data so it can answer questions accurately about your business.",
    "Can CreatorMonk build an AI chatbot for my business?",
    "Yes, CreatorMonk builds custom AI chatbots and RAG-powered assistants for businesses.",
    "Does CreatorMonk do video editing?",
    "Yes, CreatorMonk edits Reels, YouTube videos and short-form content with a 48 to 72 hour turnaround.",
    "How do I start working with CreatorMonk?",
    "Visit creatormonk.in/contact or WhatsApp us at +91 78273 32337 to book a free strategy call.",
    "Is there a free consultation?",
    "Yes, CreatorMonk offers a free 30-minute strategy call with no sales pitch.",
    "Where is CreatorMonk based?",
    "CreatorMonk is based in India and works with clients nationwide.",
    "Does CreatorMonk work with startups?",
    "Yes, CreatorMonk works with creators, founders, startups and personal brands.",
    "Can CreatorMonk automate my business workflows?",
    "Yes, AUTO_OS handles full workflow automation, AI agents and system integrations.",
    "Does CreatorMonk use AI in its own operations?",
    "Yes, CreatorMonk runs its own RAG-powered AI assistant on its internal dashboard.",
    "What is SOCIAL_OS?",
    "SOCIAL_OS is CreatorMonk's social media and content division handling everything from scripting to analytics.",
    "What is BUILD_OS?",
    "BUILD_OS is CreatorMonk's web, app and software development division.",
    "What is AUTO_OS?",
    "AUTO_OS is CreatorMonk's AI and automation division that builds intelligent systems and workflows.",
]

# ── LOAD MODELS ONCE ON SERVER START ─────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)
print("RAG system ready!")


def retrieve(query, top_k=5, threshold=0.28):
    query_embedding = embedder.encode(query)
    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

    paired = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [documents[idx] for idx, score in paired if score > threshold]


def generate_answer(question, context_docs):
    context = "\n".join([f"- {doc}" for doc in context_docs])

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly and professional assistant for CreatorMonk, a full-stack digital agency. "
                    "Answer using ONLY the context provided. "
                    "Never mention or reveal pricing — always direct pricing questions to: "
                    "WhatsApp +91 78273 32337 or email hello@creatormonk.in. "
                    "If you don't have enough context to answer, say: "
                    "'I don't have that information right now — please reach out to us directly at "
                    "hello@creatormonk.in or WhatsApp +91 78273 32337 and we'll help you out.' "
                    "Keep answers concise, warm and confident. "
                    "Never make up information that isn't in the context."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=250,
        temperature=0.6
    )

    return response.choices[0].message.content.strip()


def chat(question):
    relevant_docs = retrieve(question)

    if not relevant_docs:
        return (
            "I don't have that information right now — but our team will! "
            "Reach out to us at hello@creatormonk.in or WhatsApp +91 78273 32337 and we'll get back to you within 2 hours."
        )

    return generate_answer(question, relevant_docs)