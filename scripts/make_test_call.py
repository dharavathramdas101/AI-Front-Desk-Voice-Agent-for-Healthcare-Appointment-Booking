"""
Twilio calls YOU — run this and your phone will ring.
Usage: python scripts/make_test_call.py +91XXXXXXXXXX
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.config as config
from twilio.rest import Client

def call(to_number: str) -> None:
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    base = config.PUBLIC_BASE_URL.rstrip("/")
    call = client.calls.create(
        url=f"{base}/twilio/incoming-call",
        to=to_number,
        from_=config.TWILIO_PHONE_NUMBER,
    )
    print(f"Call initiated: {call.sid}")
    print(f"Status: {call.status}")
    print(f"Calling {to_number} from {config.TWILIO_PHONE_NUMBER}")
    print("Pick up your phone — you'll hear the greeting!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_test_call.py +91XXXXXXXXXX")
        sys.exit(1)
    call(sys.argv[1])
