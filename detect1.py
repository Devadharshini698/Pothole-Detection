"""
Firebase Integration for Pothole Logs
- Separate script to upload existing pothole logs from JSON to Firebase Firestore.
- Assumes you have a Firebase project set up with Firestore enabled.
- Install: pip install firebase-admin
- Setup: Download service account key JSON from Firebase Console > Project Settings > Service Accounts > Generate new private key.
- Replace 'path/to/serviceAccountKey.json' with your key file path.
- Collection: 'pothole_logs' in Firestore.
- Run this after detection to sync logs.
"""

import json
import firebase_admin
from firebase_admin import credentials, firestore
import os

# --- Configuration ---
SERVICE_ACCOUNT_KEY_PATH = 'potholeshack-6d5f9-firebase-adminsdk-fbsvc-1315467f66.json'  # Update with your path
PROJECT_ID = 'potholeshack-6d5f9'  # From Firebase Console
LOG_FILE = 'logs/pothole_logs.json'  # From previous detection script

def upload_logs_to_firestore(log_file, project_id):
    """Upload pothole logs to Firestore."""
    # Initialize Firebase Admin
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {'projectId': project_id})

    db = firestore.client()
    collection_ref = db.collection('pothole_logs')

    # Load logs
    if not os.path.exists(log_file):
        print(f"[Firebase] Log file {log_file} not found. Skipping upload.")
        return

    with open(log_file, 'r') as f:
        logs = json.load(f)

    batch = db.batch()
    for log in logs:
        # Use timestamp as document ID for uniqueness - convert to str
        doc_id = str(log['timestamp'])
        batch.set(collection_ref.document(doc_id), log)

    # Commit batch
    batch.commit()
    print(f"[Firebase] Uploaded {len(logs)} log entries to Firestore.")

if __name__ == "__main__":
    upload_logs_to_firestore(LOG_FILE, PROJECT_ID)