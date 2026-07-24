from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def env(key, default=None, required=False):
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


SECRET_KEY = env("DJANGO_SECRET_KEY", required=True)
DEBUG = env("DEBUG", "False").lower() == "true"
ALLOWED_HOSTS = [h.strip() for h in env("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")]

# ---- Chat / RAG ----
# Provider-agnostic: any OpenAI-compatible endpoint works.
LLM_BASE_URL = env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = env("LLM_API_KEY", required=True)
LLM_MODEL = env("LLM_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN = env("HF_TOKEN", required=True)


# ---- Email ----
RESEND_API_KEY = env("RESEND_API_KEY")
LEAD_NOTIFY_TO = env("LEAD_NOTIFY_TO", "hello@creatormonk.in")
LEAD_FROM_EMAIL = env("LEAD_FROM_EMAIL", "CreatorMonk <hello@creatormonk.in>")

# ---- Business contact ----
WHATSAPP_NUMBER = env("WHATSAPP_NUMBER", "917827332337")
CONTACT_EMAIL = env("CONTACT_EMAIL", "hello@creatormonk.in")



# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'chat',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware', 
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/6.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': env("SQLITE_PATH", BASE_DIR / 'data' / 'db.sqlite3'),
    }
}

# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = 'static/'

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://creatormonk.in",
    "https://www.creatormonk.in",
]


# DRF Rate Limiting
# REST_FRAMEWORK = {
#     'DEFAULT_THROTTLE_CLASSES': [
#         'rest_framework.throttling.AnonRateThrottle',
#     ],
#     'DEFAULT_THROTTLE_RATES': {
#         'anon': '30/hour',  # 30 requests per hour per IP
#     }
# }