# Chrome Dino Vision Autoplayer

A small end-to-end computer vision project that automatically plays the Chrome Dino game by combining a **PyTorch CNN**, **OpenCV-based obstacle localization**, and **rule-based jump control**.

This was an early Vision project built during a bootcamp before my larger robotics projects. The project is intentionally kept close to the original working implementation, with only repository structure, paths, documentation, and readability cleaned up for reproducibility.

## Project Overview

The system captures the Chrome Dino game screen in real time, detects upcoming obstacles, estimates their horizontal position, and presses the space bar when a jump is required.

```text
Game screen capture (MSS)
        ↓
Grayscale / resize
        ↓
┌─────────────────────────────┐
│ CNN obstacle classification │  → auxiliary diagnostic signal
└─────────────────────────────┘
        +
┌─────────────────────────────┐
│ OpenCV contour detection    │  → nearest obstacle x position
└─────────────────────────────┘
        ↓
Dynamic jump threshold
        ↓
Jump decision
        ↓
Keyboard input (Space)
```

## My Role

This was a small team activity that effectively became an individual implementation.

- **Dataset image collection:** teammate
- **Dataset organization / preprocessing:** me
- **CNN architecture and training pipeline:** me
- **Model evaluation and inference integration:** me
- **OpenCV obstacle localization and filtering:** me
- **Dynamic jump logic and game-control integration:** me
- **Real-time visualization / debugging UI:** me

## Key Implementation Details

### 1. CNN obstacle classifier

A small binary CNN was implemented directly in PyTorch rather than using a pretrained detector.

- Input: grayscale `256 × 64`
- Conv2d `1 → 32` + ReLU + MaxPool
- Conv2d `32 → 64` + ReLU + MaxPool
- Fully connected binary output + Sigmoid
- Loss: `BCELoss`
- Optimizer: `Adam`
- Training epochs: `20`

The recorded training run used **1,950 training images** and evaluated on **250 test images**, producing **99.60% test accuracy on that test set**.

> This number only describes the provided test split and should not be interpreted as general performance across arbitrary screen sizes, themes, or game conditions.

### 2. OpenCV obstacle localization

The CNN predicts whether an obstacle is present, but the controller also needs to know **where the nearest obstacle is** to decide when to jump.

For that reason, the final real-time control path uses OpenCV to:

- normalize day/night screen colors with binary thresholding,
- extract external contours,
- ignore the player dinosaur region,
- remove tiny noise contours,
- ignore high-flying obstacles that should not trigger a jump,
- return the x position of the nearest valid obstacle.

### 3. Dynamic jump threshold

The Chrome Dino game becomes faster over time. A fixed jump distance therefore became less reliable as gameplay continued.

The controller increases the jump threshold over elapsed game time:

```python
jump_threshold = 70 + elapsed_time * 0.5
jump_threshold = min(jump_threshold, 230)
```

This is a simple rule-based adaptation rather than a learned policy.

### 4. Real-time debugging view

The `AI Eye` window visualizes the controller's current decision state:

- **Blue line:** current jump threshold
- **Green line:** height filter for ignored obstacles
- **Red line:** nearest detected obstacle
- **CNN status:** obstacle / clear diagnostic result

The visualization was used to tune the obstacle filters and jump timing during development.

## Important Design Note

The final jump decision is based on the **OpenCV-derived obstacle position + dynamic threshold**, not directly on the CNN binary output.

The CNN remains integrated as an auxiliary obstacle-presence classifier. During development, I found that binary presence classification alone was insufficient for control because the player needs positional information to determine **when** to jump.

This distinction is preserved in the repository instead of presenting the project as a fully CNN-driven controller.

## Game State Handling

The runtime compares consecutive frames to detect long periods with almost no visual change. This is used to identify a waiting/game-over state and reset the speed timer when movement resumes.

**Automatic game restart is not implemented.** The code detects that gameplay has resumed; it does not press the restart key by itself.

## Repository Structure

```text
Chrome-Dino-Vision-Autoplayer/
├── README.md
├── requirements.txt
├── models/
│   └── dino_cnn_model.pth
├── notebooks/
│   └── train_cnn.ipynb
└── src/
    ├── dino_solver.py
    └── main.py
```

## Setup

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run

1. Open the Chrome Dino game.
2. Run:

```bash
python src/main.py
```

3. Drag over the Dino game area in the ROI selection window and press `Enter`.
4. The program clicks the selected area and starts the game with `Space`.
5. Press `q` to stop.

> Screen geometry and threshold values were tuned for the original development environment, so retuning may be necessary on a different display or browser scale.

## Limitations

This is an early learning project, and several limitations are intentionally documented rather than hidden:

- obstacle rules contain manually tuned pixel thresholds,
- the controller is specific to the Chrome Dino visual layout,
- the CNN test set is small and does not establish broad generalization,
- CNN classification is not used as the final jump gate,
- game-over is detected, but automatic restart is not implemented,
- the original dataset images are not included in this repository.

## What I Learned

This project was my first complete experience of connecting a vision model to an action loop:

**data → training → inference → image processing → decision logic → real-time action**.

Later robotics projects expanded the same perception-to-action idea using ROS 2, YOLO-based detection, robot manipulation, and NVIDIA Isaac Sim.
