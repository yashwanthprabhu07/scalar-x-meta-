# ============================================================
# integrations/gmail.py — Real Gmail Integration
# Same interface as mock EmailApp — drop-in replacement
# ============================================================

import os
import base64
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]

CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "..", "credentials.json")
TOKEN_FILE       = os.path.join(os.path.dirname(__file__), "..", "token.json")


def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


class RealEmailApp:
    """
    Real Gmail integration — same interface as mock EmailApp.
    Drop-in replacement for production use.
    """

    def __init__(self):
        self.service = get_gmail_service()
        self._sent = []

    def read_inbox(self, max_results=10):
        results = self.service.users().messages().list(
            userId="me",
            maxResults=max_results,
            labelIds=["INBOX"],
        ).execute()

        messages = results.get("messages", [])
        inbox = []
        for msg in messages:
            detail = self.service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()

            headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
            inbox.append({
                "id":      msg["id"],
                "from":    headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "date":    headers.get("Date", ""),
                "snippet": detail.get("snippet", ""),
            })

        return inbox

    def read_email(self, email_id):
        detail = self.service.users().messages().get(
            userId="me",
            id=email_id,
            format="full",
        ).execute()

        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}

        body = ""
        payload = detail.get("payload", {})
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part["body"].get("data", "")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        break
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        return {
            "id":      email_id,
            "from":    headers.get("From", ""),
            "to":      headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date":    headers.get("Date", ""),
            "body":    body,
        }

    def send_email(self, to, subject, body):
        msg = MIMEText(body)
        msg["to"]      = to
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        result = self.service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()
        self._sent.append({"to": to, "subject": subject, "id": result["id"]})
        return {"status": "sent", "message_id": result["id"], "to": to}

    def reply_email(self, email_id, body):
        original = self.read_email(email_id)
        msg = MIMEText(body)
        msg["to"]      = original["from"]
        msg["subject"] = "Re: " + original["subject"]

        detail = self.service.users().messages().get(
            userId="me", id=email_id, format="metadata",
            metadataHeaders=["Message-Id"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        thread_id = detail.get("threadId", "")

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        result = self.service.users().messages().send(
            userId="me",
            body={"raw": raw, "threadId": thread_id},
        ).execute()

        self._sent.append({"to": original["from"], "subject": msg["subject"], "id": result["id"]})
        return {
            "status":     "sent",
            "message_id": result["id"],
            "to":         original["from"],
            "reply_to":   email_id,
        }

    @property
    def sent(self):
        return self._sent