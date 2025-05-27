# Business Card Reader (BCR)

## Automating Business Card Digitization with Deep Learning

<img width="458" alt="Screenshot 2025-05-24 at 8 01 47 pm" src="https://github.com/user-attachments/assets/d3b1ac3f-32b6-476e-984f-977d97abca7d" />


The Business Card Reader (BCR) is a web-based application designed to streamline professional networking by automating the process of extracting and managing information from physical business cards. Built as a capstone project for the "Deep Learning and Convolutional Neural Network" course, BCR leverages state-of-the-art computer vision and natural language processing techniques to convert physical cards into a searchable, editable digital contact list.

---

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Project Architecture Overview](#3-project-architecture-overview)
4.  [Dataset](#4-dataset)
5.  [CNN Architecture Design](#5-cnn-architecture-design)
6.  [GUI Design](#6-gui-design)
7.  [Installation](#7-installation)
8.  [Usage](#8-usage)
9.  [Demo](#9-demo)
10. [Limitations](#10-limitations)
11. [Future Enhancements](#11-future-enhancements)
12. [Team](#12-team)
13. [License](#13-license)

---

## 1. Introduction

The project addresses the pervasive inefficiency of manually managing business cards. In an increasingly digital world, relying on physical cards for contact information leads to time-consuming data entry, potential for human error, and physical clutter. BCR provides an intelligent solution by offering a web application that can scan business cards, automatically extract crucial contact details (name, company, phone, email), and store them in a personalized digital wallet, enhancing data accuracy and accessibility.

## 2. Features

*   **User Authentication:** Simple login system to provide a personalized, secure space for each user's contacts.
*   **Live Scan & Image Upload:** Capture business card images directly via a live camera feed or upload existing image files.
*   **Automated Card Detection:** Real-time detection of business cards within the image, supporting both single and multiple cards in one frame.
*   **Information Extraction:** Advanced Optical Character Recognition (OCR) combined with Natural Language Processing (NLP) to accurately extract structured information (name, company, phone number, email) from detected cards.
*   **Manual Correction:** Intuitive interface to review and manually edit any extracted information for accuracy.
*   **Digital Card Wallet:** A dedicated "My Cards" section to view, manage, and remove saved business card contacts.
*   **Robust Data Augmentation:** Utilizes `albumentations` library to significantly expand the training dataset and improve model generalization.

## 3. Project Architecture Overview

The BCR system follows a modular architecture:

1.  **User Interface (Frontend):** A web-based application provides the user-facing interface for login, scanning, editing, and managing cards.
2.  **Backend Server:** Handles user requests, interacts with the deep learning model, processes data, and manages the database.
3.  **Object Detection Module:** Employs a trained YOLOv8 model to identify and localize business cards within captured images.
4.  **Information Extraction Pipeline:** Consists of an OCR component to convert images to text, followed by an NLP/LLM (specifically OpenAI API) to parse and categorize the extracted text into structured contact information.
5.  **Database:** A user-specific JSON-based database for storing digitized business card details.

**Workflow:**
*   **Login:** User logs in with their email/username and password.
*   **Capture Image:** User captures an image of a business card via live camera feed or uploads one.
*   **Card Detection:** The YOLOv8 model detects business cards and draws bounding boxes around them.
*   **Image Cropping & Processing:** Detected card regions are cropped and sent to the OpenAI API.
*   **Information Extraction & Classification:** The OpenAI API extracts and categorizes contact information (name, company, phone, email).
*   **Edit & Save:** Extracted information is displayed for user review and manual correction before being saved to the user's personal database.
*   **View & Manage:** Users can access their saved contacts in the "My Cards" section.

## 4. Dataset

The dataset primarily consists of business card images collected and annotated by the team.

*   **Initial Collection:** A custom dataset was manually compiled by photographing various business cards from team members' contacts. The initial goal was over 200 images, though approximately 54 served as the core base.
*   **Annotation:** All images were meticulously annotated with bounding boxes for cards and specific entities using CVAT, with annotations stored in **YOLO format (.txt files)**.
*   **Preprocessing:** Images were resized (640x640 or 1024x1024 pixels) and subjected to grayscale conversion and normalization.
*   **Data Augmentation:** To significantly enhance model robustness and generalizability, extensive data augmentation was performed using the `albumentations` library. This pipeline generates a large volume of additional training images (targeting over 3000 total images from the base 54) through transformations like horizontal flips, random brightness/contrast, rotations, shifts, and scaling. While external Kaggle datasets were explored, the primary focus for dataset expansion remained on this robust augmentation strategy.

## 5. CNN Architecture Design

We employed the **YOLOv8** object detection framework with a **CSPDarknet** backbone. Through experimentation with three CNN configurations, we aimed to optimize performance:

*   **CNN-1 (Baseline):** Input 640x640, no augmentation, confidence threshold 0.6.
*   **CNN-2 (Augmented):** Input 640x640, built-in YOLO augmentation, confidence threshold 0.6.
*   **CNN-3 (High-Resolution + Fine-Tuning) - Final Chosen Architecture:**
    *   **Input Size:** Increased to 1024x1024 for capturing finer details.
    *   **Augmentation:** Extensive data augmentation using `albumentations`.
    *   **Confidence Threshold:** Reduced from 0.6 to **0.3** to improve sensitivity for detecting more potential objects, including small or partially obscured cards.
    *   **Purpose:** This configuration was chosen for its ability to capture all business cards while maintaining acceptable accuracy, enhancing small-object detection accuracy and overall model sensitivity.

## 6. GUI Design

The application's GUI is a web-based prototype, organized into four main components:

*   **Login Page:** Provides email/username and password fields with "Login" and "Sign Up" buttons for user authentication.
*   **Scanning Page:** Displays a live camera feed for capturing business card images. Includes "Capture Image" and "Logout" buttons. This page also presents the detected cards with bounding boxes and confidence scores.
*   **Edit Page:** After detection, extracted information (Name, Company Name, Phone Number, Email) is pre-filled into editable text fields, alongside a small image of the card. Users can manually correct any errors and save changes.
*   **My Cards Page (Results Page):** Displays a list of all saved business cards, each with its image and extracted details. This acts as the user's digital contact wallet.

*(For detailed screenshots of each page, please refer to the project report's "2.3 GUI Design" section.)*

## 7. Installation

To set up and run the BCR application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd BusinessCardReader
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` should contain `torch`, `torchvision`, `ultralytics` (for YOLOv8), `opencv-python`, `albumentations`, `python-dotenv` (for API keys), and any web framework dependencies like `Flask` or `Django`.)*

4.  **Set up OpenAI API Key:**
    *   Obtain an API key from OpenAI.
    *   Create a `.env` file in the project root directory.
    *   Add your API key to the `.env` file:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ```

5.  **Prepare Dataset (if running training locally):**
    *   Ensure your `dataset/images` and `dataset/labels` directories are populated with your base images and YOLO-formatted annotations.
    *   Run the data augmentation script (e.g., `albumentations_augmentation.py`) to generate augmented data into `augmented_dataset/images/train` and `augmented_dataset/labels/train`.
    *   (Details on specific training scripts would go here if training is part of the local setup.)

## 8. Usage

1.  **Start the application (Backend & Frontend):**
    ```bash
    python app.py # Or your main application entry point
    ```
    *(Note: The `app.py` or equivalent would handle starting the web server. If using a specific framework like Flask, it might be `flask run`.)*

2.  **Access the application:**
    *   Open your web browser and navigate to the application's local address (e.g., `http://127.0.0.1:5000`).
    *   If using `ngrok` for public exposure (as seen in demo), use the provided `ngrok` URL.

3.  **Login/Sign Up:**
    *   On the login page, use an existing account or create a new one (username/email and password).

4.  **Scan/Upload Business Card:**
    *   Navigate to the scanning page.
    *   Use the live camera feed to capture a business card or upload an image.
    *   Confirm the detected card(s).

5.  **Edit Details:**
    *   Review the extracted information on the "Edit Card Details" page.
    *   Make any necessary manual corrections.
    *   Click "Save Changes" or "Save All" to store the card in your personal digital wallet.

6.  **View My Cards:**
    *   Go to the "My Cards" section to view and manage your saved business cards.

## 9. Demo

A live demonstration of the application's functionality is available in the provided video:

*   **[Link to your demo video]** (e.g., `https://example.com/bcr_demo.mp4`)

The demo showcases the full user journey from login, live scanning and detection, automatic information extraction, manual editing, and finally, saving and viewing contacts in the digital wallet.

## 10. Limitations

*   **Angle Sensitivity:** Model accuracy decreases with images captured from highly unconventional or aggressive angles, indicating a need for more diverse angular training data.
*   **Background Interference:** Performance degrades when business cards closely match background colors or textures, sometimes leading to missed detections or inaccurate cropping.
*   **False Positives:** The model occasionally misclassifies other rectangular objects (e.g., remote controls, phones) as business cards.
*   **Limited Entity Extraction:** Currently extracts only core contact details (name, company, phone, email); does not extract job titles, full addresses, or social media links.
*   **Dataset Diversity:** Despite augmentation, the relatively small original custom dataset might limit generalizability to a wider variety of global business card designs and scripts not encountered during training.
*   **Database Scalability:** The JSON-based database, while functional for individual users, may not be suitable for large-scale production environments with extensive user bases and data volumes.

## 11. Future Enhancements

*   **Expanded & Diversified Dataset:** Integrate larger public datasets and implement more advanced data collection methods to improve model generalization.
*   **Enhanced Object Detection:** Explore more robust detection techniques or ensemble methods for challenging conditions and reduced false positives.
*   **Comprehensive Information Extraction:** Expand entity recognition to include job titles, full addresses, and social media links. Improve multilingual OCR and NLP capabilities.
*   **Scalable Backend & Database:** Transition to a more robust database solution (e.g., SQL or NoSQL) and implement a full-fledged user authentication system for better scalability and security.
*   **Additional Features:** Implement batch card processing, QR code scanning, direct actions (call, email, navigate), and integration with external CRM or contact management platforms.
*   **Performance Optimization:** Further optimize image processing, model inference speed, and overall application responsiveness.

