# Facial Recognition Attendance System (FRAS)

## Overview

The Facial Recognition Attendance System (FRAS) is a project developed to enhance the efficiency and accuracy of attendance tracking in educational institutions using facial recognition technology. The system utilizes Convolutional Neural Networks (CNNs) to recognize and verify individuals' faces and manage attendance records automatically.

## Description

The FRAS project aims to replace traditional manual attendance systems with an automated solution that leverages facial recognition. This system addresses common issues associated with manual attendance, such as time wastage and inaccuracy.

### Features

- **Facial Recognition:** Uses CNNs to identify and verify individuals.
- **Automated Attendance Tracking:** Automatically records attendance based on facial recognition.
- **User-Friendly Interface:** Provides a simple interface for administrators to manage and view attendance records.
- **Real-Time Processing:** Processes and records attendance in real-time.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Flask (for web interface)

### Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/FRAS.git
    cd FRAS
    ```

2. **Create a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download and configure the pre-trained models (if applicable).**

5. **Run the Flask web server (if applicable):**

    ```sh
    python app.py
    ```

## Usage

1. **Start the application:**

    Run the Flask server using the command provided in the installation section.

2. **Access the web interface:**

    Open a web browser and navigate to `http://localhost:5000` (or the URL provided by your Flask server).

3. **Upload images or use the webcam:**

    Use the interface to either upload images of individuals or use a webcam for real-time attendance tracking.

4. **Manage and view attendance records:**

    Navigate through the interface to manage and view attendance records.

## Code Overview

- `app.py`: Main entry point for the Flask web server.
- `model.py`: Contains the implementation of the facial recognition model using CNNs.
- `utils.py`: Utility functions for image processing and data handling.
- `templates/`: Contains HTML templates for the web interface.
- `static/`: Contains static files such as CSS and JavaScript.

## Results

- **Projected Improvement:** The system is projected to improve attendance tracking efficiency by approximately 20% based on technical analysis and performance metrics.
- **Dissertation:** A detailed dissertation is available that provides actionable recommendations for educational institutions and discusses ethical considerations for technology integration.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. For major changes, please open an issue to discuss the changes before submitting a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Contributors:** Thank you to everyone who has contributed to this project.
- **Libraries and Tools:** TensorFlow, OpenCV, Flask, and other libraries used in this project.

## Contact

For any questions or inquiries, please contact:

- **Name:** Sumit Singh
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/sumit-singh-282044166)
- **GitHub:** [GitHub Profile](https://github.com/sumitsingh28871)
