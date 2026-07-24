from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services import chat
from .throttles import ChatRateThrottle

MAX_HISTORY = 12


class ChatView(APIView):
    throttle_classes = [ChatRateThrottle]

    def post(self, request):
        question = (request.data.get("question") or "").strip()

        if not question:
            return Response({"error": "No question provided"},
                            status=status.HTTP_400_BAD_REQUEST)

        if len(question) > 1000:
            return Response({"error": "Question too long"},
                            status=status.HTTP_400_BAD_REQUEST)

        # Client sends the recent turns — keeps the server stateless
        raw = request.data.get("history") or []
        history = []
        if isinstance(raw, list):
            for turn in raw[-MAX_HISTORY:]:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role")
                text = turn.get("text")
                if role in ("user", "bot") and isinstance(text, str):
                    history.append({"role": role, "text": text[:1000]})

        return Response(chat(question, history), status=status.HTTP_200_OK)