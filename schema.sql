CREATE DATABASE IF NOT EXISTS face_recognition_db;
USE face_recognition_db;

-- embedding: 128 float64 values, little-endian, 1024 bytes
-- (rows from version 1 were pickled; run `python -m faceverify.cli migrate` once)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding BLOB NOT NULL
);
