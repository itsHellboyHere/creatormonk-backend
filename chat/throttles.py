from rest_framework.throttling import AnonRateThrottle
from rest_framework.exceptions import Throttled

class ChatRateThrottle(AnonRateThrottle):
    rate = '30/hour'

    def throttle_failure(self):
        raise Throttled(detail={
            "error": "Too many requests",
            "message": "You've reached the limit of 30 messages per hour. Please try again later! 🙏",
            "type": "rate_limit_exceeded"
        })