CREATE DATABASE IF NOT EXISTS face_recognition_db;
USE face_recognition_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding BLOB NOT NULL
);
