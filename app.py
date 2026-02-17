import face_recognition  # Library for face recognition
import os  # Library to handle file paths and directories
import numpy as np  # Library for numerical operations
import pickle # Library to handle binary serialization
from flask import Flask, request, jsonify, render_template # Flask web framework
import mysql.connector # Library for MySQL database

app = Flask(__name__, template_folder='home')

# --- Database Config ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",      # Default XAMPP password is empty
        database="face_recognition_db"
    )

# --- Helper Functions ---
def convert_embedding_to_bytes(embedding):
    """Converts numpy array embedding to bytes for storage."""
    return pickle.dumps(embedding)

def convert_bytes_to_embedding(embedding_bytes):
    """Converts bytes back to numpy array."""
    return pickle.loads(embedding_bytes)

# --- Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enroll', methods=['POST'])
def enroll_user():
    # 1) Enroll a user
    # Input: name, image
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing name or image"}), 400
    
    name = request.form['name']
    image_file = request.files['image']

    try:
        # User behavior: Detect one face in the image.
        image = face_recognition.load_image_file(image_file)  # Load image
        
        # User behavior: Extract the face embedding.
        # Ensure only one face is present for enrollment
        encodings = face_recognition.face_encodings(image)  # Extract face encodings
        
        if len(encodings) == 0:
            return jsonify({"error": "No face detected"}), 400
        if len(encodings) > 1:
             return jsonify({"error": "Multiple faces detected. Please upload an image with a single face."}), 400
        
        known_face_encoding = encodings[0] # Get the first face encoding
        embedding_bytes = convert_embedding_to_bytes(known_face_encoding)

        # User behavior: Store name and embedding in MySQL.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, embedding) VALUES (%s, %s)", (name, embedding_bytes))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        return jsonify({"message": f"User {name} enrolled successfully", "id": user_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_user():
    # 2) Verify a face
    if 'image' not in request.files:
        return jsonify({"error": "Missing image"}), 400
    
    image_file = request.files['image']

    try:
        # User behavior: Detect one face.
        image = face_recognition.load_image_file(image_file)  # Load input image
        face_locations = face_recognition.face_locations(image)  # Detect faces in the image
        
        if len(face_locations) == 0:
            return jsonify({"error": "No face detected"}), 400
            
        # User behavior: Extract embedding.
        # Extract face encodings from the detected faces
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Taking the first face found for verification
        face_encoding = face_encodings[0]

        # User behavior: Load all embeddings from the database.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, embedding FROM users")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
             return jsonify({"message": "No users in database", "match": False}), 404

        # Prepare lists to store match data
        known_face_encodings = []
        known_face_names = []

        for row in rows:
            known_face_encodings.append(convert_bytes_to_embedding(row[2]))
            known_face_names.append(row[1])

        # User behavior: Compare using face_recognition.compare_faces().
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # Compare detected face with known faces
        name = "Unknown"
        match_found = False

        # Compute distances to known faces and find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)  # Find the closest match

        if matches[best_match_index]:  # If a match is found
            name = known_face_names[best_match_index]
            match_found = True
            
        # Return match result and name if found.
        if match_found:
            return jsonify({"message": "User verified", "name": name, "match": True}), 200
        else:
            return jsonify({"message": "User not found", "match": False}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users', methods=['GET'])
def list_users():
    # 3) List users
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # User behavior: Return all users. Do not return embeddings.
        cursor.execute("SELECT id, name FROM users")
        users = cursor.fetchall()
        conn.close()
        
        # Format the response
        user_list = [{"id": u[0], "name": u[1]} for u in users]
        return jsonify(user_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    # 4) Delete a user
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # User behavior: Remove the user from the database.
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        
        rows_deleted = cursor.rowcount
        conn.close()

        if rows_deleted == 0:
            return jsonify({"error": "User not found"}), 404
            
        # Return confirmation message.
        return jsonify({"message": f"User {user_id} deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
