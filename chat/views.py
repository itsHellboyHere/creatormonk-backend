from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services import chat
from .throttles import ChatRateThrottle


class ChatView(APIView):
    throttle_classes = [ChatRateThrottle]

    def post(self, request):
        question = (request.data.get("question") or "").strip()

        if not question:
            return Response(
                {"error": "No question provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(question) > 1000:
            return Response(
                {"error": "Question too long"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # chat() already returns {"answer", "language", "grounded"} —
        # don't wrap it again.
        return Response(chat(question), status=status.HTTP_200_OK)