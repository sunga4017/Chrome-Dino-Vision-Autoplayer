import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

from dino_solver import predict_logic


pyautogui.PAUSE = 0

# Runtime constants retained from the original working prototype.
Y_LIMIT_AI = 35
INITIAL_JUMP_THRESHOLD = 70
THRESHOLD_GROWTH_PER_SEC = 0.5
MAX_JUMP_THRESHOLD = 230
NO_OBSTACLE_X = 999


def select_game_region():
    """Let the user select the Chrome Dino game region from a screenshot."""

    print("1. Capturing the full screen...")
    with mss.mss() as sct:
        monitor_full = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor_full))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    print("2. Drag over the Dino game area and press ENTER.")
    rect = cv2.selectROI("Select Game Area", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = rect
    if w == 0:
        return None

    return {
        "top": y + monitor_full["top"],
        "left": x + monitor_full["left"],
        "width": w,
        "height": h,
    }


def main():
    monitor = select_game_region()
    if monitor is None:
        return

    print("=== Dino Vision Autoplayer Ready ===")
    time.sleep(2)

    cx = monitor["left"] + monitor["width"] // 2
    cy = monitor["top"] + monitor["height"] // 2
    pyautogui.click(cx, cy)

    # Start the game.
    keyboard.press("space")
    time.sleep(0.1)
    keyboard.release("space")

    start_time = time.time()
    prev_frame_gray = None
    is_game_running = True
    death_duration = 0

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed("q"):
                print("Program terminated")
                break

            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect long periods of almost no frame change (game-over/waiting state).
            if prev_frame_gray is not None:
                score = np.mean(cv2.absdiff(gray, prev_frame_gray))
                if score < 1.0:
                    death_duration += 1
                    if death_duration > 100:
                        is_game_running = False
                else:
                    if not is_game_running:
                        print("Restart detected: resetting speed timer")
                        start_time = time.time()
                        is_game_running = True
                    death_duration = 0

            prev_frame_gray = gray.copy()

            # Perception: CNN presence classification + OpenCV obstacle localization.
            has_obstacle, obs_x = predict_logic(frame)

            # Increase the jump distance as the game speeds up.
            elapsed = time.time() - start_time
            jump_threshold = INITIAL_JUMP_THRESHOLD + (
                elapsed * THRESHOLD_GROWTH_PER_SEC
            )
            jump_threshold = min(jump_threshold, MAX_JUMP_THRESHOLD)

            # Final action decision uses the localized obstacle position.
            should_jump = (obs_x < jump_threshold) and (obs_x != NO_OBSTACLE_X)

            h_real, w_real, _ = frame.shape
            scale_x = w_real / 256.0
            scale_y = h_real / 64.0

            # Blue: current jump threshold.
            real_threshold_x = int(jump_threshold * scale_x)
            cv2.line(
                frame,
                (real_threshold_x, 0),
                (real_threshold_x, h_real),
                (255, 0, 0),
                2,
            )

            # Green: height filter.
            real_limit_y = int(Y_LIMIT_AI * scale_y)
            cv2.line(
                frame,
                (0, real_limit_y),
                (w_real, real_limit_y),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Ignore Height < {Y_LIMIT_AI}",
                (max(10, w_real - 240), max(20, real_limit_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            # CNN result is displayed as a diagnostic signal; it does not gate jumping.
            cnn_text = "CNN: OBSTACLE" if has_obstacle else "CNN: CLEAR"
            cv2.putText(
                frame,
                cnn_text,
                (10, max(55, h_real - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            # Red: nearest valid obstacle.
            if obs_x != NO_OBSTACLE_X:
                real_obs_x = int(obs_x * scale_x)
                cv2.line(
                    frame,
                    (real_obs_x, 0),
                    (real_obs_x, h_real),
                    (0, 0, 255),
                    2,
                )

                status = "RUNNING" if is_game_running else "WAITING"
                text_color = (0, 255, 0) if is_game_running else (0, 255, 255)
                info_text = (
                    f"[{status}] Dist: {obs_x} < Limit: {int(jump_threshold)}"
                )
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )

            cv2.imshow("AI Eye", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if should_jump and is_game_running:
                print(f"JUMP (distance: {obs_x})")
                keyboard.press("space")
                time.sleep(0.05)
                keyboard.release("space")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
