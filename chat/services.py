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
    # BRAND IDENTITY — ENGLISH Style
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
    "CreatorMonk does not just sell services — it builds systems that generate revenue.",
    "CreatorMonk replaces manual work with AI and automation.",
    "CreatorMonk focuses on lead generation, conversion, automation and scaling.",
    "CreatorMonk's approach is: Build, Grow, Automate, Scale.",
    "CreatorMonk is a complete business growth engine, not just a service provider.",
    "CreatorMonk helps businesses generate leads, close sales and automate operations end-to-end.",

    # BRAND IDENTITY — HINDI
    "CreatorMonk ek full-stack digital agency hai jo 2025 mein founded hui hai.",
    "CreatorMonk India mein based hai aur poore desh ke clients ke saath kaam karta hai.",
    "CreatorMonk creators, founders, startups aur personal brands ko digitally grow karne mein help karta hai.",
    "CreatorMonk teen divisions mein kaam karta hai: SOCIAL_OS, BUILD_OS aur AUTO_OS.",
    "CreatorMonk sirf services nahi bechta — woh revenue generate karne wale systems banata hai.",
    "CreatorMonk ka approach hai: Build karo, Grow karo, Automate karo, Scale karo.",
    "CreatorMonk ek complete business growth engine hai, sirf ek service provider nahi.",

    # ─────────────────────────────────────────
    # CONTACT INFORMATION — ENGLISH
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

    # CONTACT INFORMATION — HINDI / HINGLISH
    "Pricing ke liye CreatorMonk ko WhatsApp karo +91 78273 32337 pe.",
    "Project discuss karne ke liye hello@creatormonk.in pe email karo.",
    "CreatorMonk sabhi inquiries ka reply 2 ghante ke andar deta hai.",
    "CreatorMonk ke working hours hain Monday se Saturday, subah 10 baje se raat 8 baje IST.",
    "Aap CreatorMonk ke saath ek free 30-minute strategy call book kar sakte ho.",
    "Free consultation call mein koi sales pitch nahi hoti — sirf honest advice milti hai.",
    "Project shuru karne ke liye creatormonk.in/contact visit karo.",
    "Onboarding mein lagbhag 3 se 5 business days lagte hain initial call ke baad.",
    "CreatorMonk har client ke liye custom strategy banata hai — koi template nahi.",

    # ─────────────────────────────────────────
    # SOCIAL_OS — ENGLISH
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
    "CreatorMonk builds Instagram DM automation systems.",
    "CreatorMonk builds lead qualification bots that work 24/7 like a human sales agent.",

    # SOCIAL_OS — HINDI / HINGLISH
    "SOCIAL_OS CreatorMonk ka social media aur content division hai.",
    "CreatorMonk Instagram accounts professionally manage karta hai.",
    "CreatorMonk Reels banata aur edit karta hai — short-form videos ke liye.",
    "CreatorMonk YouTube channels manage karta hai — strategy se lekar uploads tak.",
    "CreatorMonk LinkedIn pe personal branding karta hai.",
    "CreatorMonk monthly content calendars banata hai.",
    "CreatorMonk trending audios aur formats research karta hai har hafte.",
    "CreatorMonk monthly analytics aur performance reports deta hai.",
    "CreatorMonk Meta aur Google pe paid ads chalata hai.",
    "SOCIAL_OS ke pricing ke liye +91 78273 32337 ya hello@creatormonk.in pe contact karo.",

    # ─────────────────────────────────────────
    # BUILD_OS — ENGLISH
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
    "CreatorMonk builds high-converting landing pages and sales funnels.",
    "CreatorMonk's funnels follow the flow: offer → landing page → conversion.",
    "CreatorMonk builds CRM systems and lead tracking dashboards.",

    # BUILD_OS — HINDI / HINGLISH
    "BUILD_OS CreatorMonk ka web, app aur software development division hai.",
    "CreatorMonk Next.js se fast aur modern websites banata hai.",
    "CreatorMonk mobile apps banata hai iOS aur Android ke liye React Native mein.",
    "CreatorMonk SaaS platforms banata hai — authentication, billing aur dashboards ke saath.",
    "CreatorMonk e-commerce websites aur custom shopping portals banata hai.",
    "CreatorMonk har project ke baad 30 din ka post-launch support deta hai.",
    "CreatorMonk pehle UI/UX design karta hai, phir code likhta hai.",
    "CreatorMonk high-converting landing pages aur sales funnels banata hai.",
    "CreatorMonk payment gateways aur third-party APIs integrate karta hai.",
    "BUILD_OS ke pricing ke liye +91 78273 32337 ya hello@creatormonk.in pe contact karo.",

    # ─────────────────────────────────────────
    # AUTO_OS — ENGLISH
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
    "CreatorMonk builds customer support and sales closing bots.",

    # AUTO_OS — HINDI / HINGLISH
    "AUTO_OS CreatorMonk ka AI aur automation division hai.",
    "CreatorMonk custom AI agents banata hai jo large language models se powered hote hain.",
    "CreatorMonk RAG systems banata hai jo company ke apne data pe train hote hain.",
    "RAG system ek aisa AI assistant hai jo aapki company ke baare mein sab kuch jaanta hai.",
    "CreatorMonk AI chatbots banata hai customer support aur sales ke liye.",
    "CreatorMonk business workflows automate karta hai n8n aur custom pipelines se.",
    "CreatorMonk manual repetitive kaam ko automation se khatam karta hai.",
    "CreatorMonk ke AI systems 24 ghante, 7 din bina manual input ke kaam karte hain.",
    "AUTO_OS ke pricing ke liye +91 78273 32337 ya hello@creatormonk.in pe contact karo.",

    # ─────────────────────────────────────────
    # WHATSAPP AUTOMATION — ENGLISH
    # ─────────────────────────────────────────
    "CreatorMonk offers WhatsApp automation and bulk messaging services.",
    "CreatorMonk sends bulk WhatsApp messages for cold outreach, lead nurturing and promotions.",
    "CreatorMonk builds WhatsApp sales systems with auto-reply and AI conversation handling.",
    "CreatorMonk builds appointment booking and follow-up sequences on WhatsApp.",
    "CreatorMonk builds outbound sales automation that replaces a manual sales team.",
    "CreatorMonk automates cold lead targeting with script-based messaging on WhatsApp.",
    "CreatorMonk integrates WhatsApp automation with sales funnels and conversion tracking.",
    "CreatorMonk can send thousands of personalized WhatsApp messages automatically.",

    # WHATSAPP AUTOMATION — HINDI / HINGLISH
    "CreatorMonk WhatsApp automation aur bulk messaging services offer karta hai.",
    "CreatorMonk bulk WhatsApp messages bhejta hai cold outreach, lead nurturing aur promotions ke liye.",
    "CreatorMonk WhatsApp sales systems banata hai — auto-reply aur AI conversation handling ke saath.",
    "CreatorMonk WhatsApp pe appointment booking aur follow-up sequences set karta hai.",
    "CreatorMonk outbound sales automation banata hai jo manual sales team ki jagah kaam karta hai.",
    "CreatorMonk hazaaron personalized WhatsApp messages automatically bhej sakta hai.",

    # ─────────────────────────────────────────
    # CREATIVE PRODUCTION — ENGLISH
    # ─────────────────────────────────────────
    "CreatorMonk produces ad videos and brand promotional videos.",
    "CreatorMonk creates business jingles in both audio and video format.",
    "CreatorMonk creates short-form ad creatives and high-converting ad scripts.",

    # CREATIVE PRODUCTION — HINDI / HINGLISH
    "CreatorMonk ad videos aur brand promotional videos banata hai.",
    "CreatorMonk business jingles create karta hai audio aur video format mein.",
    "CreatorMonk short-form ad creatives aur high-converting ad scripts banata hai.",

    # ─────────────────────────────────────────
    # TECH STACK
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
    "CreatorMonk uses n8n for automation workflows.",

    # ─────────────────────────────────────────
    # PROCESS — ENGLISH
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

    # PROCESS — HINDI / HINGLISH
    "CreatorMonk ka process ek free discovery call se shuru hota hai.",
    "Discovery call ke baad CreatorMonk ek custom strategy banata hai.",
    "CreatorMonk development shuru karne se pehle designs aur wireframes present karta hai.",
    "CreatorMonk agile sprints mein kaam karta hai aur har hafte client ko update deta hai.",
    "CreatorMonk social media kaam shuru karne se pehle full brand audit karta hai.",
    "Video editing ka turnaround time 48 se 72 ghante hai.",
    "CreatorMonk launch se pehle saare systems ko stress-test karta hai.",

    # ─────────────────────────────────────────
    # FAQ — ENGLISH
    # ─────────────────────────────────────────
    "What does CreatorMonk do?",
    "CreatorMonk is a full-stack digital agency offering social media management, web and software development, AI automation, WhatsApp automation, sales funnels and paid advertising.",
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
    "Does CreatorMonk do WhatsApp marketing?",
    "Yes, CreatorMonk builds WhatsApp automation systems including bulk messaging, auto-replies and AI-powered sales bots.",
    "Can CreatorMonk build a sales bot?",
    "Yes, CreatorMonk builds complete sales conversation systems including cold outreach, lead qualification and closing flows.",
    "Does CreatorMonk build sales funnels?",
    "Yes, CreatorMonk builds high-converting landing pages and sales funnels integrated with ads and CRMs.",
    "Can CreatorMonk replace my sales team?",
    "CreatorMonk builds AI and WhatsApp automation systems that handle lead qualification, follow-ups and closing automatically.",

    # ─────────────────────────────────────────
    # FAQ — HINDI / HINGLISH
    # ─────────────────────────────────────────
    "CreatorMonk kya karta hai?",
    "CreatorMonk ek full-stack digital agency hai jo social media management, website development, AI automation, WhatsApp automation, sales funnels aur paid ads ki services deta hai.",
    "CreatorMonk ke saath kaam kaise shuru karein?",
    "creatormonk.in/contact visit karo ya +91 78273 32337 pe WhatsApp karo — free strategy call book karo.",
    "Kya CreatorMonk WhatsApp marketing karta hai?",
    "Haan, CreatorMonk WhatsApp automation systems banata hai — bulk messaging, auto-reply aur AI-powered sales bots ke saath.",
    "Kya CreatorMonk mobile app banata hai?",
    "Haan, CreatorMonk iOS aur Android apps banata hai React Native mein.",
    "Kya CreatorMonk meri business ki automation kar sakta hai?",
    "Haan, AUTO_OS division full workflow automation, AI agents aur system integrations handle karta hai.",
    "Kya CreatorMonk ke saath free consultation milti hai?",
    "Haan, CreatorMonk ek free 30-minute strategy call offer karta hai jisme koi sales pitch nahi hoti.",
    "CreatorMonk ki pricing kya hai?",
    "Pricing ke liye +91 78273 32337 pe WhatsApp karo ya hello@creatormonk.in pe email karo.",
    "Kya CreatorMonk sales funnel banata hai?",
    "Haan, CreatorMonk high-converting landing pages aur sales funnels banata hai jo ads aur CRMs se connected hote hain.",
    "Kya CreatorMonk AI chatbot bana sakta hai?",
    "Haan, CreatorMonk custom AI chatbots aur RAG-powered assistants banata hai businesses ke liye.",
    "CreatorMonk kitne time mein reply karta hai?",
    "CreatorMonk sabhi inquiries ka reply 2 ghante ke andar deta hai.",
    "Onboarding mein kitna time lagta hai?",
    "Onboarding mein lagbhag 3 se 5 business days lagte hain initial call ke baad.",
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

                    "LANGUAGE RULE — this is very important: "
                    "1. If the user's question is in Hindi (Devanagari script), reply fully in Hindi using Devanagari script. "
                    "2. If the user's question is in Hinglish (Hindi words written in English letters, or a mix of Hindi and English), reply in Hinglish — casual, warm, energetic and natural. "
                    "3. If the user's question is in English, reply in English. "
                    "Always match the user's language and tone exactly. "

                    "PRICING RULE: Never mention or reveal any pricing. "
                    "Always direct pricing questions to: WhatsApp +91 78273 32337 or email hello@creatormonk.in. "

                    "FALLBACK RULE: If you don't have enough context to answer, respond in the user's language as follows — "
                    "English: 'I don't have that information right now — please reach out to us at hello@creatormonk.in or WhatsApp +91 78273 32337 and we'll get back to you within 2 hours.' "
                    "Hinglish: 'Yeh information abhi mere paas nahi hai — +91 78273 32337 pe WhatsApp karo ya hello@creatormonk.in pe email karo, 2 ghante mein reply milega!' "
                    "Hindi: 'यह जानकारी अभी मेरे पास नहीं है — कृपया +91 78273 32337 पर WhatsApp करें या hello@creatormonk.in पर ईमेल करें, हम 2 घंटे में जवाब देंगे।' "

                    "Keep answers concise, warm and confident. "
                    "Never make up information that isn't in the context."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def chat(question):
    relevant_docs = retrieve(question)

    if not relevant_docs:
        return (
            "I don't have that information right now — but our team will! "
            "Reach out to us at hello@creatormonk.in or WhatsApp +91 78273 32337 "
            "and we'll get back to you within 2 hours."
        )

    return generate_answer(question, relevant_docs)