from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import chat

class ChatView(APIView):
    def post(self,request):
        question = request.data.get("question","")
        if not question:
            return Response(
                {"error":"No question provided"},
                status=status.HTTP_400_BAD_REQUEST
            )
        answer = chat(question)
        return Response({"answer":answer},status=status.HTTP_200_OK)

