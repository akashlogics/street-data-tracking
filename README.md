# Street People Tracking System

This Python project tracks people moving across predefined street zones using video footage. It classifies the individuals by gender, determines their mode of movement (walking or cycling), and logs the data into an Excel file for further analysis. The output video highlights tracked individuals with bounding boxes and labels.

## Features

- **Real-time object detection:** Uses YOLOv8 for detecting people and bicycles in the video.
- **Classification:** Classifies individuals as "Man" or "Woman" based on their aspect ratio.
- **Mode of movement:** Differentiates between walking and cycling.
- **Street zone tracking:** Tracks individuals as they move through predefined street boundaries.
- **Data logging:** Records timestamped data (gender, mode, source street, destination street) in an Excel file.
- **Output video:** Annotates the video with bounding boxes, movement trails, and labels for individuals.

## Libraries Used

The following Python libraries are used in this project:

- **[OpenCV](https://opencv.org/):** For video processing, drawing bounding boxes, and displaying output frames.
- **[YOLOv8](https://github.com/ultralytics/ultralytics):** For object detection.
- **[OpenPyXL](https://openpyxl.readthedocs.io/):** For creating and managing the Excel file to store tracking data.
- **[NumPy](https://numpy.org/):** For numerical operations.
- **[datetime](https://docs.python.org/3/library/datetime.html):** To handle timestamps for tracked events.
- **[collections (deque)](https://docs.python.org/3/library/collections.html):** To store and manage movement trails.

## Algorithm Overview

1. **Video Initialization:**
   - The video is loaded using OpenCV.
   - Frame dimensions and other properties like FPS are retrieved.

2. **Street Zone Setup:**
   - The street is divided into predefined zones with vertical boundaries.

3. **Object Detection:**
   - YOLOv8 detects objects in each frame.
   - Filters detections to focus on people and bicycles.

4. **Classification:**
   - Determines the gender of each detected person based on their aspect ratio.
   - Identifies if a person is cycling by comparing their position to detected bicycles.

5. **Tracking:**
   - Each detected person is assigned a unique ID.
   - Tracks their movement across frames using bounding boxes and trails.

6. **Data Logging:**
   - Records timestamped data (gender, mode of movement, and street transition) into an Excel file.

7. **Visualization:**
   - Draws bounding boxes, trails, and street labels on the video output.
   - Annotates the video with classification and movement information.

8. **Output:**
   - Saves an annotated video file.
   - Exports an Excel file (`people_tracking_data.xlsx`) containing the tracking data.

## Installation

To run this project, ensure you have Python installed and set up the necessary dependencies:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install the required libraries:
   ```bash
   pip install opencv-python-headless ultralytics openpyxl numpy
   ```

## Usage

1. Replace the video source (`street_camera.mp4`) with your input video file in the project directory.
2. Run the script:
   ```bash
   python src5.py
   ```
3. The processed video will be saved as `output_gender_tracking.mp4`.
4. The Excel data will be saved as `people_tracking_data.xlsx`.

## Output

- **Video:** Annotated with bounding boxes, movement trails, and classifications.
- **Excel:** Contains rows with the following columns:
  - Timestamp
  - Gender
  - Mode (Walking/Cycling)
  - From Street
  - To Street

## Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to fork the repository and create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- YOLOv8 for robust object detection.
- OpenCV for video processing and visualization.
```