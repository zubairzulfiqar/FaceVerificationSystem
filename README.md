# Face Verification System

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Initialize the database:
    - Import `schema.sql` into your MySQL server.

## Running the App
1.  Run the application:
    ```bash
    python app.py
    ```
2.  Open your browser and navigate to:
    `http://127.0.0.1:5000`

## Features
-   **Enroll**: Add a new user with a face image.
-   **Verify**: Check if an uploaded face matches a user in the database.
-   **Manage**: View and delete enrolled users.
