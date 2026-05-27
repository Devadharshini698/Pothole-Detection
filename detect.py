"""
Optimized Live Pothole Detection using YOLOv8 - GPS polling runs in a separate thread to prevent lag.
- Detections are batched and logged only when moving > 50m.
- Logs a single entry per batch with highest confidence + unique pothole count.
- Uses model.track() to identify unique potholes.
- Logs are written to disk only ONCE on exit.
- Added: Incremental upload of new generated logs to Firebase Firestore on each batch commit (pip install firebase-admin). Full logs saved to JSON on exit.
"""

import cv2
from ultralytics import YOLO
import numpy as np
import geocoder  
import json
import os
import time
from datetime import datetime
import threading
import math
import firebase_admin
from firebase_admin import credentials, firestore

# --- Configuration ---
MODEL_PATH = 'pothole_training/pothole_yolov8n3/weights/best.pt'
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'pothole_logs.json')
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.25
# % of frame area for severity
FRAME_AREA_THRESHOLDS = {'low': 0.10, 'med': 0.30}
# Green, Yellow, Red
COLORS = {'low': (0, 255, 0), 'med': (0, 255, 255), 'high': (0, 0, 255)}
# Distance in meters to trigger a new log entry
LOGGING_DISTANCE_THRESHOLD = 50  # meters
# How often the GPS thread checks for a new location
GPS_POLL_INTERVAL = 5  # seconds
# Firebase Configuration
SERVICE_ACCOUNT_KEY_PATH = 'potholeshack-6d5f9-firebase-adminsdk-fbsvc-1315467f66.json'  # Update with your path
PROJECT_ID = 'potholeshack-6d5f9'  # From Firebase Console

# --- GPS Polling Thread ---
class GpsPoller(threading.Thread):
    """
    Runs GPS polling in a separate thread to avoid blocking the main loop.
    Uses IP-based geocoding (low accuracy, but demonstrates the pattern).
    For real-world use, replace with a serial/USB GPS reader (e.g., gpsd-py).
    """
    def __init__(self):
        super().__init__()
        self.daemon = True  # Thread will exit when main program exits
        self._lock = threading.Lock()
        self._running = True
        self._location = (0.0, 0.0)  # Default/mock location
        self._last_check_time = 0

    def run(self):
        print("[GPS Thread] Started.")
        while self._running:
            if time.time() - self._last_check_time > GPS_POLL_INTERVAL:
                try:
                    g = geocoder.ip('me')
                    if g.ok:
                        with self._lock:
                            self._location = g.latlng  # (lat, lon)
                        # print(f"[GPS Thread] Location updated: {self._location}")
                    else:
                        # Keep using last known location if fetch fails
                        pass
                except Exception as e:
                    print(f"[GPS Thread] Error: {e}")
                self._last_check_time = time.time()
            time.sleep(0.5)
        print("[GPS Thread] Stopped.")

    def get_location(self):
        with self._lock:
            return self._location

    def stop(self):
        self._running = False

# --- Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_logs(file_path):
    """Load existing logs from JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {file_path}. Starting fresh.")
            return []
    return []

def save_logs_to_file(file_path, logs):
    """Save all logs to JSON file (called once on exit)."""
    print(f"\n[Main] Saving {len(logs)} log entries to {file_path}...")
    with open(file_path, 'w') as f:
        json.dump(logs, f, indent=4)
    print("[Main] Save complete.")

def upload_single_log_to_firestore(entry, project_id):
    """Upload a single new generated log entry to Firestore immediately after batch commit."""
    # Initialize Firebase Admin if not already
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred, {'projectId': project_id})
        except Exception as e:
            print(f"[Firebase] Error initializing: {e}. Skipping upload. Check SERVICE_ACCOUNT_KEY_PATH.")
            return

    db = firestore.client()
    collection_ref = db.collection('pothole_logs')

    if not entry:
        print("[Firebase] No entry to upload.")
        return

    # Use timestamp as document ID for uniqueness - convert to str
    doc_id = str(entry['timestamp'])
    try:
        db.collection('pothole_logs').document(doc_id).set(entry)
        print(f"[Firebase] Uploaded single log entry (timestamp: {doc_id}) to Firestore.")
    except Exception as e:
        print(f"[Firebase] Error uploading single entry: {e}")

def commit_log_batch(batch, location, logs_list):
    """ Processes a batch of detections, creates a single log entry, appends to local list, and uploads to Firebase. """
    if not batch:
        return
    # Find the single best detection (highest confidence) in the batch
    best_detection = max(batch, key=lambda d: d['conf'])
    # Find the number of unique potholes (using track_id)
    unique_pothole_ids = set(d['id'] for d in batch)
    num_unique_potholes = len(unique_pothole_ids)
    timestamp = time.time()
    lat, lon = location
    entry = {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp).isoformat(),
        "lat": lat,
        "lon": lon,
        "severity": best_detection['severity'],
        "confidence": float(best_detection['conf']),
        "num_unique_potholes": num_unique_potholes
    }
    logs_list.append(entry)
    print(f"[Log Commit] Logged {num_unique_potholes} potholes at {(lat, lon)}. Best conf: {best_detection['conf']:.2f}")
    # Immediately upload the new generated entry to Firebase
    upload_single_log_to_firestore(entry, PROJECT_ID)

def get_severity(box, frame_area):
    """Classify severity based on bounding box area."""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    bbox_area = (x2 - x1) * (y2 - y1)
    area_ratio = bbox_area / frame_area
    if area_ratio < FRAME_AREA_THRESHOLDS['low']:
        severity = 'low (surface_crack)'
        color = COLORS['low']
    elif area_ratio < FRAME_AREA_THRESHOLDS['med']:
        severity = 'med (pit)'
        color = COLORS['med']
    else:
        severity = 'high (depression)'
        color = COLORS['high']
    return severity, color, (x1, y1, x2, y2)

# --- Main Detection Loop ---
def main():
    # Load model
    print("[Main] Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    # Start camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height

    # Start GPS thread
    gps_poller = GpsPoller()
    gps_poller.start()

    # Load logs
    os.makedirs(LOG_DIR, exist_ok=True)
    all_logs = load_logs(LOG_FILE)

    # Logging state variables
    last_log_location = gps_poller.get_location()
    current_detection_batch = []  # Stores detections for the current 50m area

    print("\n[Main] Live detection started.")
    print("New logs will be pushed to Firebase on each batch; full logs saved to JSON on exit.")
    print("Controls: Press 'q' to quit, 's' to save frame.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] Failed to grab frame. Exiting.")
                break

            # Get current GPS location (fast, non-blocking read)
            current_gps = gps_poller.get_location()

            # Run tracking
            # We use .track() to get a unique 'box.id' for each pothole
            results = model.track(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, persist=True)
            annotated_frame = frame.copy()  # Start with a clean frame

            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                # Calculate distance from last log point
                distance_moved = haversine(last_log_location[0], last_log_location[1], current_gps[0], current_gps[1])

                # --- Batching Logic ---
                # If we've moved more than 50m, commit the last batch and start a new one
                if distance_moved > LOGGING_DISTANCE_THRESHOLD:
                    print(f"[Main] Moved {distance_moved:.1f}m. Committing log batch...")
                    commit_log_batch(current_detection_batch, last_log_location, all_logs)
                    # Reset for new batch
                    current_detection_batch = []
                    last_log_location = current_gps

                # --- Process and Batch Detections ---
                for box in detections:
                    # Check if box.id is present (it might not be on the first frame)
                    if box.id is None:
                        continue
                    track_id = int(box.id[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0:  # Pothole class
                        severity, color, (x1, y1, x2, y2) = get_severity(box, frame_area)
                        # Add detection info to the current batch
                        current_detection_batch.append({
                            'id': track_id,
                            'conf': conf,
                            'severity': severity
                        })
                        # Draw custom box + label
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"Pothole (ID: {track_id}) {severity}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display info
            info_text = f"GPS: {current_gps[0]:.4f}, {current_gps[1]:.4f} | Logs: {len(all_logs)} | Batch: {len(current_detection_batch)}"
            cv2.putText(annotated_frame, info_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Optimized Pothole Detection', annotated_frame)

            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Main] 'q' pressed. Shutting down.")
                break
            elif key == ord('s'):
                cv2.imwrite(f'detection_{time.time()}.jpg', annotated_frame)
                print("[Main] Frame saved!")

    finally:
        # --- Cleanup and Final Save ---
        print("[Main] Cleaning up...")
        # Stop GPS thread
        gps_poller.stop()
        gps_poller.join()

        # Commit the final batch of detections (pushes to Firebase if any)
        print("[Main] Committing final log batch...")
        commit_log_batch(current_detection_batch, last_log_location, all_logs)

        # Save all logs to file (appends new generated data)
        save_logs_to_file(LOG_FILE, all_logs)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] Detection stopped. Check logs for flagged locations.")

if __name__ == "__main__":
    main()