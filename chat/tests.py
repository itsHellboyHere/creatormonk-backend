from django.test import TestCase
from rest_framework.test import APIClient
from unittest.mock import patch

class ChatAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch('chat.views.chat')
    def test_chat_returns_answer(self, mock_chat):
        mock_chat.return_value = "CreatorMonk is a creator agency"
        response = self.client.post('/api/chat/',
            {'question': 'what is creatormonk?'},
            format='json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('answer', response.data)

    def test_empty_question_returns_400(self):
        response = self.client.post('/api/chat/',
            {'question': ''},
            format='json'
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_question_returns_400(self):
        response = self.client.post('/api/chat/',
            {},
            format='json'
        )
        self.assertEqual(response.status_code, 400)