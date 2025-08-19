# PenBot Software

![PenBot drawing a portrait](https://github.com/user-attachments/assets/35ec1e40-f3d7-4ee3-b102-978ef88c18da)

Welcome to the official repository for the software that powers **PenBot**, a linkage-based drawing robot that turns human faces into unique vector portraits. This repo contains all the janky but functional Python code that runs on the built-in hardware. This workflow uses computer vision, low-level motor control, coordinate processing, and more.

The project was born from a simple, unconventional idea: to have a robot sign my senior yearbook. It has since evolved into a complex and engaging project. For a full breakdown of the mechanical design, project evolution, and real-world testing (including its appearance at OpenSauce!), please visit the official project portfolio page:

**[Dorian Todd - PenBot Project Page](https://www.doriantodd.com/projects/penbot)**

---

## Features

*   **Real-time Face to Vector Pipeline**: Captures a live video feed, detects a face, and converts it into a stylized, single-line vector portrait in real-time.
*   **Custom Machine Code**: Converts the generated SVG vector art into a simple, custom `.pb` command file optimized for the robot's specific kinematics.
*   **Advanced Motion Control**: Implements inverse kinematics to translate Cartesian coordinates into complex motor angles. A Digital Differential Analyzer (DDA) algorithm ensures smooth, coordinated movement between the two coaxial stepper motors.
*   **Interactive UI**: An OpenCV-based user interface provides a live preview, capture controls, and real-time drawing progress.
*   **Modular Architecture**: The code is separated into logical components for handling vision, SVG conversion, robot driving, and motion control.

<img width="972" height="783" alt="penbot-topdown" src="https://github.com/user-attachments/assets/da2453ae-0724-438c-b44f-2a393414124f" />

## Software Architecture

The software operates as a multi-stage pipeline, orchestrating the journey from a live image to a physical drawing.

1.  **Capture (`face_capture.py`)**:
    *   Uses `OpenCV` to access the webcam.
    *   Employs `MediaPipe` for robust face detection and background segmentation.
    *   Applies a series of image processing techniques (blur, adaptive threshold, skeletonization via `scikit-image`) to create a stylized line art representation.
    *   The final contours are saved as a plotter-friendly SVG file.

2.  **Convert (`svg_to_pb.py`)**:
    *   The generated `face.svg` is parsed using `svgpathtools`.
    *   Paths are broken down into a series of small, linear segments to ensure drawing accuracy.
    *   These segments are translated into a simple machine language for the bot:
        *   `PBM,delay,x,y` - **M**ove pen to canvas coordinate (x,y).
        *   `PBZ,steps` - Move **Z**-axis (pen lift/lower).
        *   `PBR,state` - **R**elease (0) or enable (1) motors.

3.  **Execute (`robot_driver.py` & `penbot_movement.py`)**:
    *   The `robot_driver` reads the `.pb` file line by line.
    *   For each `PBM` command, it calls the `penbot_movement` module.
    *   The movement module is the core of the motion system, containing the inverse kinematics and the DDA algorithm for smooth pathing.

## The Core Mathematics: Inverse Kinematics

The most significant contribution of this codebase is the solution to the inverse kinematics problem for PenBot's unique linkage system. The value of this repository lies primarily in the math and calculations required to translate a desired pen position on the canvas into precise angles for the two coaxial motors.

These calculations are contained within **`penbot_math.py`**.

To make this complex geometry easier to understand, we developed an interactive simulation that visualizes the trigonometric relationships and constraints of the linkage. This is an excellent resource for anyone looking to understand how the robot's coordinate system works.

**[Interactive Linkage Simulation on Desmos](https://www.desmos.com/calculator/yp5hnid8xg)**
<img width="1158" height="593" alt="image" src="https://github.com/user-attachments/assets/a3eb785f-2424-4890-856e-b4d18436c5e7" />

## Running the OpenSauce Demo

The main entry point, `main.py`, launches the interactive portrait-drawing application exactly as it was used to draw over 50 portraits for attendees at OpenSauce 2025.

### Prerequisites

*   A Raspberry Pi with Raspberry Pi OS (or a similar Debian-based Linux system).
*   Python 3.x and the dependencies listed in the "Getting Started" section below.
*   I2C enabled on the Raspberry Pi (`sudo raspi-config`).
*   The specific PenBot hardware, including two Motor HATs at I2C addresses `0x61` and `0x62`.
> Note: I highly reccomend looking into arduino based stepper motor controll such as the arduino mega cnc-controller, as the Adafruit stepper motor hats can only realistically power 2 stepper motors, and do not have room for over-volting or speed tuning.

### Running the Application

1.  Position the robot's pen at its physical home position on the canvas.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  The OpenCV window will launch. Follow the on-screen prompts to capture and draw a face.

> [!WARNING]
> **This software is a hardware-specific demonstration, not a general-purpose tool.**
>
> This script is tightly coupled to the PenBot hardware configuration. It contains hardcoded values for motor addresses, canvas dimensions, and robot kinematics located in `penbot_constants.py`. There are no built-in features to adapt this code to different hardware.
>
> Furthermore, the application has known stability issues and may crash unexpectedly. It is provided as-is to demonstrate the complete system in action and to serve as a reference for the underlying mathematics.
**[Twitch streamer crashing the program live...](https://youtu.be/rEQlkDJpYmw?si=KWCmzqyQXtPSB5B3&t=615)**


## Getting Started (for development)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/penbot-software.git
    cd penbot-software
    ```

2.  Install the required Python libraries. You will need to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install opencv-python numpy mediapipe scikit-image svgpathtools svgelements adafruit-circuitpython-motorkit
    ```

## Work in Progress

This repository represents the state of the software as used for its primary goal—signing yearbooks and drawing portraits at OpenSauce. It is very much a work in progress.

**Known Issues & Areas for Improvement:**
*   **Code Cleanup**: The code is functional but could benefit from additional refactoring, more extensive comments, and better error handling.
*   **Calibration Routine**: The calibration process currently involves manually editing `penbot_constants.py`. A guided calibration script would be a major improvement.
*   **Performance**: The software pipeline can be slow, especially the face-to-vector processing.
*   **Hardware Robustness**: As noted in the portfolio, the robot can be sensitive to power fluctuations and may experience overheating during long sessions.

Contributions, bug reports, and suggestions are welcome! Please feel free to open an issue or submit a pull request. Also feel free to build your own better version of this linkage, I would love to see it!

## Acknowledgements and Credits

PenBot stands on the shoulders of giants and is a direct result of the generous sharing of knowledge within the maker community. I want to extend special thanks to the projects that provided both the initial inspiration and the foundational blueprint for this robot.

### The Line-us Drawing Robot

The initial spark for this project came from the brilliant [Line-us](https://www.line-us.com/), a compact, network-connected drawing arm that successfully Kickstarted in 2017. Their elegant design and engaging concept demonstrated the potential for small, artistic robotic arms and served as the primary inspiration for PenBot.

### The Line-us Clone by Barton Dring

In the early stages, this project was heavily guided by the fantastic work of Barton Dring, who documented his own [Line-us clone on his buildlog.net blog](https://www.buildlog.net/blog/2017/02/a-line-us-clone/). This article was an invaluable resource that helped kickstart the development process.

<img width="300" height="260" alt="line-us-clone" src="https://github.com/user-attachments/assets/325f3dcc-2ade-4ccc-9a91-d1663e1dcac3" />

His detailed post provided the foundational blueprint, including the reverse-engineered arm lengths and, most importantly, the fundamental approach to solving the inverse kinematics using intersecting circles. This documentation was the critical starting point that enabled the development of PenBot’s own mathematical models and control software.

### Project Contributors

Of course, PenBot would not have been possible without the direct contributions of the core team!

*   **Wandian Lee**: Developed the foundational inverse kinematics mathematics and the original C-based control code that proved the concept.
*   **James Todd**: Refactored the control logic from C to Python and developed much of the current software architecture running on the Raspberry Pi.

Finally, a huge thank you to the developers and communities behind Python, OpenCV, MediaPipe, Adafruit, and the many other open-source libraries that made this project possible.
