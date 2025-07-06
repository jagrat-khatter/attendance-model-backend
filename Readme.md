# 🧠 Deep Learning-Based Attendance Recognition Backend

This is the **Python Flask backend** for a full-stack **AI-powered contactless attendance system**. It performs **facial recognition** using the **FaceNet** model (via DeepFace) and returns the verified identity of users based on real-time webcam images.

---

## 🔧 What This Backend Does

- Accepts an image file via a POST request (`/recognize`)
- Generates a **128-dimensional face embedding**
- Compares the embedding with preloaded reference embeddings (from `dataset/`)
- Returns the closest match and cosine similarity score
- Sends match result to another backend (Node.js) to **log attendance**

---

## 📁 Project Structure

Model-Backend/
│
├── app.py # Main Flask app (API endpoints)
├── requirements.txt # Python dependencies
├── venv/ # (Optional) Python virtual environment
├── models/ # (Place downloaded model here)
├── dataset/ # Folder containing known user face images
├── uploads/ # Temporary uploaded images
├── static/, templates/ # (Optional) For any UI or styling
├── utils.py # (Optional) Helper scripts
└── README.md # You are here.

yaml
Copy
Edit

---

## 🚀 Getting Started (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-python-backend-repo.git
cd your-python-backend-repo
2. 🐍 Set Up Python Environment (Recommended)
Use a virtual environment to manage dependencies cleanly:

bash
Copy
Edit
python -m venv venv
Activate it:

Windows

bash
Copy
Edit
venv\Scripts\activate
macOS/Linux

bash
Copy
Edit
source venv/bin/activate
3. 📦 Install Required Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. 📁 Download Pretrained Model (FaceNet)
GitHub doesn't allow uploading files >100MB, so you’ll need to manually download the FaceNet model:

Download from: [Dropbox Link] or [Google Drive Mirror] (add the actual link)

Place the .h5 file here:

bash
Copy
Edit
Model-Backend/models/facenet512.h5
5. 🖼️ Add Face Dataset
Structure your dataset as:

Copy
Edit
dataset/
├── Alice_101/
│   ├── 1.jpg
│   └── 2.jpg
├── Bob_102/
│   └── face.png
Each subfolder = one person. Images inside are used to preload embeddings.

6. ▶️ Run the Flask Server
bash
Copy
Edit
python app.py
Your server will be available at:

arduino
Copy
Edit
http://localhost:5000
7. 🌍 Expose Server Using Ngrok
To connect your frontend to this backend:

bash
Copy
Edit
ngrok http 5000
This will give a public HTTPS link like:

arduino
Copy
Edit
ngrok http --domain=allowing-tapir-remarkably.ngrok-free.app 5000
🔬 How It Works
On startup, the server caches embeddings from all dataset images.

When a request is made to /recognize, it:

Generates an embedding from the incoming image

Compares it to each cached embedding using cosine similarity

Picks the best match if similarity is below the threshold (e.g., 0.25)

The matched identity and distance are returned as JSON

📦 Sample API Usage
Endpoint: /recognize
Method: POST (multipart/form-data)

Request Body:

Key: image

Value: Image file (.jpg, .png)

Response:

json
Copy
Edit
{
  "match": true,
  "person": "Alice 101",
  "distance": 0.182
}
🧪 Notes
Tested on 12 unique individuals

Accuracy held up across lighting and background changes

Backend is not deployed due to large model size and high compute needs

🤝 Linked Repositories
🔗 Frontend (React + TensorFlow.js): https://github.com/jagrat-khatter/attendance-frontend

🔗 Node.js Attendance Logger: https://github.com/jagrat-khatter/attendance-node-backend

🧠 Requirements (Sample requirements.txt)
nginx
Copy
Edit
flask
flask-cors
deepface
numpy
scipy
tensorflow
📍 TODO (Optional Enhancements)
 Add Dockerfile for easy deployment

 Add model auto-download script

 Add image upload + registration endpoint

🙏 Acknowledgements
DeepFace

FaceNet

Flask

Ngrok

markdown
Copy
Edit
