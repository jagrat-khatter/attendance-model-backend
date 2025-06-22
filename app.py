import os
from flask import Flask, request, jsonify
from deepface import DeepFace
from scipy.spatial.distance import cosine
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache: stores both valid and invalid embeddings
representations_cache = []

def preload_embeddings():
    print("[INFO] Preloading face embeddings from dataset...")
    for person_name in os.listdir(DATASET_FOLDER):
        person_dir = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False  # ⬅️ Allow all images
                )

                if not reps or "embedding" not in reps[0]:
                    print(f"[⚠️] No embedding found for {img_path}")
                    embedding = None
                else:
                    embedding = np.array(reps[0]["embedding"])
                    print(f"[✅] Cached embedding for {img_path}")

            except Exception as e:
                print(f"[❌] Error in {img_path}: {e}")
                embedding = None

            representations_cache.append({
                "embedding": embedding,
                "identity": person_name,
                "path": img_path
            })

    print(f"[📦] Total cached images: {len(representations_cache)}")

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)
    print(f"[📸] Uploaded image saved to: {image_path}")

    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )

        if not reps or "embedding" not in reps[0]:
            print(f"[⚠️] No face embedding found in uploaded image.")
            return jsonify({
                "match": False,
                "person": "Unknown",
                "distance": None
            })

        uploaded_rep = np.array(reps[0]["embedding"])

        best_match = None
        lowest_distance = 1.0  # cosine similarity max
        for cached in representations_cache:
            if cached["embedding"] is None:
                continue  # skip unusable
            dist = cosine(uploaded_rep, cached["embedding"])
            if dist < lowest_distance:
                lowest_distance = dist
                best_match = cached["identity"]

        THRESHOLD = 0.25
        if lowest_distance <= THRESHOLD:
            print(f"[🟢] Match: {best_match}, Distance: {lowest_distance:.4f}")
            return jsonify({
                "match": True,
                "person": best_match,
                "distance": float(lowest_distance)
            })
        else:
            print(f"[🔴] No match found. Closest: {best_match}, Distance: {lowest_distance:.4f}")
            return jsonify({
                "match": False,
                "person": "Unknown",
                "distance": float(lowest_distance)
            })

    except Exception as e:
        print(f"[❌] Recognition failed: {e}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route("/mark-attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    name = data.get("name")
    distance = data.get("distance")

    print(f"[📝] Attendance marked for: {name} | Distance: {distance}")
    return jsonify({"message": f"Attendance marked for {name}"}), 200

if __name__ == "__main__":
    preload_embeddings()
    app.run(debug=True, host="0.0.0.0", port=5000)