"""
CreatorMonk knowledge base.

Rules that keep this useful:
  - Every fact must match what's actually on creatormonk.in
  - No pricing, ever. Pricing goes to a call.
  - Each entry is one self-contained fact (better retrieval than paragraphs)
  - Hindi entries are real Hindi, not translated English word-for-word
"""

# ══════════════════════════════════════════════════════════
#  WHO WE ARE
# ══════════════════════════════════════════════════════════
IDENTITY = [
    "CreatorMonk is a web, AI and automation agency based in Greater Noida, Uttar Pradesh, India.",
    "CreatorMonk builds websites, mobile apps, AI assistants, WhatsApp automation, branding and social media content.",
    "CreatorMonk started by making YouTube thumbnails, then video edits, then social media and websites, then WhatsApp and AI bots, and now full mobile apps.",
    "CreatorMonk is run by three founders: Rohan Raj handles tech and growth, Vishal Kumar handles software and AI, and Kundan Choudhary handles studio and operations.",
    "When you work with CreatorMonk you talk directly to the founders. There are no account managers in between.",
    "CreatorMonk works with clients across India, mostly in Delhi NCR, Noida and Greater Noida.",
    "CreatorMonk gives one clear quote after a short call. There are no surprise charges later.",
    "CreatorMonk hands over everything at the end: your code, your domain, your accounts and your logins. Nothing stays locked behind the agency.",
    "CreatorMonk prefers launching something real and improving it, over polishing privately for months.",
    "CreatorMonk sends regular updates even when there is nothing exciting to report, so clients never have to chase for status.",

    # Hindi
    "CreatorMonk ek web, AI aur automation agency hai jo Greater Noida, Uttar Pradesh mein hai.",
    "CreatorMonk websites, mobile apps, AI assistants, WhatsApp automation, branding aur social media content banata hai.",
    "CreatorMonk ki shuruaat YouTube thumbnails banane se hui thi, phir video editing, phir social media aur websites, phir WhatsApp aur AI bots, aur ab poore mobile apps.",
    "CreatorMonk teen founders chalate hain. Client seedha founders se baat karta hai, beech mein koi account manager nahi hota.",
    "CreatorMonk ek short call ke baad ek clear quote deta hai. Baad mein koi chhupa hua charge nahi aata.",
    "Project khatam hone par sab kuch client ko mil jaata hai — code, domain, accounts aur logins. Kuch bhi agency ke paas lock nahi rehta.",
]

# ══════════════════════════════════════════════════════════
#  SERVICES — mirrors /services on the site
# ══════════════════════════════════════════════════════════
SERVICES = [
    # --- Websites ---
    "CreatorMonk builds custom websites and web software that load fast and work on every screen size.",
    "CreatorMonk websites are built to show up on Google search and to turn visitors into customers.",
    "CreatorMonk gives you a short training after launch so you can change text and images on your website yourself, without a developer.",
    "Every CreatorMonk website includes a contact form, a WhatsApp button and Google Analytics setup.",
    "CreatorMonk gives 3 months of free support and small updates after a website goes live.",
    "CreatorMonk sets up your domain and hosting for you, and you own all of it.",
    "If you already have a website, CreatorMonk will review it and honestly tell you what to keep, fix or rebuild.",
    "Most CreatorMonk websites are ready in a few weeks. A clear timeline is shared after the first call.",
    "Websites CreatorMonk has built include Unidecor, Rocky Shakti Agro Science, Srishti Solar Power, Radha Krishna Bio Plantic, Navyya Nirman and Hexalam.",

    # --- Apps ---
    "CreatorMonk builds mobile apps that work on both iPhone and Android.",
    "CreatorMonk app projects include user login, profiles, push notifications, an admin panel and the backend.",
    "CreatorMonk designs every app screen and gets your approval before writing any code.",
    "CreatorMonk publishes your finished app to the App Store and Play Store under your own account.",
    "CreatorMonk gives free bug fixes after an app launches.",

    # --- AI & automation ---
    "CreatorMonk builds AI assistants and chatbots that reply to customers 24 hours a day, including on WhatsApp.",
    "CreatorMonk AI bots are trained on your own business information, so they sound like your brand and not like generic AI.",
    "CreatorMonk AI bots speak Hindi, English and Hinglish, and regional languages are possible too.",
    "If a CreatorMonk AI bot is unsure about an answer, it hands the conversation over to a real person instead of guessing.",
    "CreatorMonk connects AI automation to tools businesses already use, like WhatsApp, Instagram, Gmail, Google Sheets and CRMs.",
    "CreatorMonk builds WhatsApp drip campaigns that follow up with leads automatically.",
    "CreatorMonk automation handles repeat work so a team can spend time on real work instead.",
    "Simple AI bots from CreatorMonk go live in a couple of weeks. Bigger systems take longer.",
    "Customer data used by CreatorMonk AI systems stays private and is only used to answer that business's customers.",

    # --- Social media ---
    "CreatorMonk manages social media end to end: planning, designing, posting, and replying to comments and DMs.",
    "A CreatorMonk social media month includes a content plan, 12 or more posts, 8 or more reels, custom captions and a simple monthly report.",
    "CreatorMonk covers Instagram, YouTube Shorts, LinkedIn and Facebook.",
    "CreatorMonk clients approve every social media post before it goes live.",
    "CreatorMonk does not promise fake follower numbers. It promises consistent, quality content.",
    "CreatorMonk suggests starting social media with 3 months so results have time to show, then month to month.",
    "Clients own all social media content CreatorMonk creates — designs, videos and source files.",

    # --- Branding ---
    "CreatorMonk does full branding: logo with variations, brand colours, a font system and a simple brand book.",
    "CreatorMonk branding includes business card, letterhead, invoice and social media template designs.",
    "CreatorMonk shows 3 different logo directions first, then refines the one you pick.",
    "Clients get all brand source files from CreatorMonk in Figma, PDF, PNG and SVG.",
    "CreatorMonk can do a simple brand refresh or a complete rebrand.",

    # --- Video ---
    "CreatorMonk edits reels, YouTube shorts, long-form YouTube videos and brand films.",
    "CreatorMonk video edits include colour grading, subtitles in Hindi or English, trending music, and text and motion graphics.",
    "CreatorMonk delivers video in every size needed — vertical for Instagram, wide for YouTube, and square.",
    "CreatorMonk reels are usually ready in a couple of days.",
    "CreatorMonk includes 2 full rounds of changes on video edits.",
    "CreatorMonk only edits video and does not shoot it, but can recommend good local videographers.",

    # Hindi
    "CreatorMonk aisi websites banata hai jo tezi se khulti hain aur har screen par sahi dikhti hain.",
    "Website launch hone ke baad CreatorMonk 3 mahine tak free support deta hai.",
    "CreatorMonk mobile apps banata hai jo iPhone aur Android dono par chalti hain.",
    "CreatorMonk AI chatbots banata hai jo 24 ghante customers ko reply karte hain, WhatsApp par bhi.",
    "CreatorMonk ke AI bots Hindi, English aur Hinglish teeno mein baat karte hain.",
    "CreatorMonk WhatsApp automation aur drip campaigns banata hai jo leads ko apne aap follow up karte hain.",
    "CreatorMonk social media poora sambhalta hai — planning, design, posting aur comments ke reply.",
    "CreatorMonk branding karta hai — logo, brand colours, fonts, business card aur brand book.",
    "CreatorMonk reels, YouTube videos aur brand films edit karta hai. Reels aam taur par do din mein ready ho jaati hain.",
    "CreatorMonk sirf video edit karta hai, shooting nahi karta.",
]

# ══════════════════════════════════════════════════════════
#  HOW WE WORK
# ══════════════════════════════════════════════════════════
PROCESS = [
    "Working with CreatorMonk starts with a short call where you explain what you need. No jargon and no pressure.",
    "After the first call CreatorMonk shares a clear quote and a timeline.",
    "CreatorMonk shows you designs and gets your approval before building anything.",
    "CreatorMonk sends progress updates during a project so you always know what is happening.",
    "CreatorMonk tests everything across devices and browsers before launch.",
    "CreatorMonk stays available after launch for support and small fixes.",

    # Hindi
    "CreatorMonk ke saath kaam ek chhoti si call se shuru hota hai jisme aap apni zarurat batate hain.",
    "Pehli call ke baad CreatorMonk ek clear quote aur timeline share karta hai.",
    "CreatorMonk kuch bhi banane se pehle design dikhata hai aur aapki approval leta hai.",
    "Project ke dauraan CreatorMonk regular updates deta hai.",
]

# ══════════════════════════════════════════════════════════
#  CONTACT
# ══════════════════════════════════════════════════════════
CONTACT = [
    "You can reach CreatorMonk on WhatsApp at +91 78273 32337.",
    "You can email CreatorMonk at hello@creatormonk.in.",
    "You can start a project by visiting creatormonk.in/contact and filling the short form.",
    "CreatorMonk usually replies within a few hours during working hours.",
    "The first call with CreatorMonk is free and has no sales pitch, just honest advice.",
    "CreatorMonk is based in Greater Noida, Uttar Pradesh.",

    # Hindi
    "CreatorMonk se WhatsApp par baat karne ke liye +91 78273 32337 par message karein.",
    "CreatorMonk ko email karne ke liye hello@creatormonk.in par likhein.",
    "Project shuru karne ke liye creatormonk.in/contact par jaakar chhota sa form bharein.",
    "CreatorMonk working hours mein kuch ghanton ke andar reply kar deta hai.",
    "Pehli call bilkul free hoti hai aur usme koi sales pitch nahi hoti, sirf honest salaah milti hai.",
]

# ══════════════════════════════════════════════════════════
#  PRICING — every one of these routes to a call
# ══════════════════════════════════════════════════════════
PRICING = [
    "CreatorMonk does not list prices publicly because every project is different. You get one clear quote after a short call.",
    "To know the cost of a website, app, AI bot, branding or social media work, message CreatorMonk on WhatsApp at +91 78273 32337 or email hello@creatormonk.in.",
    "CreatorMonk pricing depends on what you need. Share your requirement on a short call and you will get a clear number with nothing hidden.",

    # Hindi
    "CreatorMonk apni pricing website par nahi likhta kyunki har project alag hota hai. Ek chhoti call ke baad ek clear quote mil jaata hai.",
    "Kisi bhi kaam ka price jaanne ke liye +91 78273 32337 par WhatsApp karein ya hello@creatormonk.in par email karein.",
    "Cost aapki zarurat par depend karta hai. Ek short call par apni requirement batayein, aapko ek saaf number mil jayega jisme kuch chhupa nahi hoga.",
]

DOCUMENTS = IDENTITY + SERVICES + PROCESS + CONTACT + PRICING