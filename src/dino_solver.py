from pathlib import Path

import cv2
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "dino_cnn_model.pth"


class CNN(torch.nn.Module):
    """Binary classifier used to detect whether an obstacle is present."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(64 * 16 * 64, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.sigmoid(out)


model = CNN().to(DEVICE)
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: {MODEL_PATH.name} ({DEVICE})")
except Exception as exc:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}") from exc


def get_real_obstacle_distance(img_gray: np.ndarray) -> int:
    """Return the x coordinate of the nearest valid obstacle, or 999 if none."""

    # Day/night normalization: keep obstacles white on a black background.
    bg_color = img_gray[0, 0]
    if bg_color > 127:
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_x = 999
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # x <= 65: player dinosaur area
        # <= 2 px: small noise
        if w > 2 and h > 2 and x > 65:
            # Ignore high-flying pterodactyls that should not trigger a jump.
            if (y + h) < 35:
                continue

            if x < min_x:
                min_x = x

    return min_x


def predict_logic(img_bgr: np.ndarray) -> tuple[bool, int]:
    """
    Run the perception pipeline.

    Returns:
        has_obstacle: CNN-based binary obstacle prediction.
        obstacle_x: nearest obstacle x position from OpenCV contour detection.
    """

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (256, 64))

    # CNN: auxiliary obstacle-presence classifier.
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        has_obstacle = output.item() > 0.8

    # OpenCV: provides the position needed for jump timing.
    obstacle_x = get_real_obstacle_distance(img_resized)

    return has_obstacle, obstacle_x
