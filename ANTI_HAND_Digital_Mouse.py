# Kullanılan kütüphaneler
import tkinter as tk
from tkinter import Tk, Label, Button, Text, Scrollbar, END, Frame
from PIL import Image, ImageTk
import threading
import queue
import sys
import os
import cv2
import mediapipe as mp
import pyautogui
import pydirectinput
import numpy as np
from collections import deque
import time
import win32gui
import win32con
import speech_recognition as sr
import pyperclip

# PyAutoGUI ve Pydirectinput ayarları
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0

# Çıktıları yakalamak için kuyruk
output_queue = queue.Queue()


# Çıktıları metin alanına yazan iş parçacığı
def update_output_text():
    try:
        while not output_queue.empty():
            message = output_queue.get_nowait()
            output_text.insert(END, message)
            output_text.see(END)
        root.after(100, update_output_text)
    except queue.Empty:
        root.after(100, update_output_text)


# EyeMouse fonksiyonu
def run_eye_mouse():
    output_queue.put("Starting EyeMouse...\n")

    # Mediapipe yüz mesh modelini başlatma
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.7,
                                                min_tracking_confidence=0.7)

    # Ekran boyutlarını alma
    screen_width, screen_height = pyautogui.size()

    # Web kamerasını başlatma
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Göz takibi için landmark indisleri
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Yumuşatma ayarları
    smooth_factor = 0.5
    speed_factor = 0.6
    mouse_x_history = deque(maxlen=100)
    mouse_y_history = deque(maxlen=100)
    prev_mouse_x, prev_mouse_y = screen_width / 2, screen_height / 2
    MOVEMENT_THRESHOLD = 10

    # Göz kırpma ayarları
    EAR_THRESHOLD = 0.25
    BLINK_DURATION = 3
    left_blink_counter = 0
    right_blink_counter = 0
    left_click_active = False
    right_click_active = False
    double_left_blink_counter = 0
    double_blink_time_window = 0.5
    last_left_blink_time = 0
    last_double_click_time = 0
    double_click_window = 0.5
    click_count = 0
    SCROLL_THRESHOLD = 0.7
    SCROLL_AMOUNT = 300
    last_blink_time = 0
    blink_sequence = []
    scroll_cooldown = 0
    pending_click = None
    click_stabilization_cooldown = 0
    CLICK_STABILIZATION_DURATION = 15
    BOTH_EYES_CLOSED_DURATION = 2.0
    both_eyes_closed_start = None

    # Sanal klavye ayarları
    keyboard_visible = False
    keyboard_window = None
    typed_text = ""
    caps_lock = False
    keyboard_buttons = []
    keyboard_layout = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
        ['space', 'enter', 'delete', 'capslock']
    ]
    focused_input = None

    def set_window_topmost(window_name):
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    def calculate_ear(eye_points, landmarks, frame_width, frame_height):
        v1 = np.linalg.norm(np.array([landmarks[eye_points[1]].x * frame_width,
                                      landmarks[eye_points[1]].y * frame_height])
                            - np.array([landmarks[eye_points[5]].x * frame_width,
                                        landmarks[eye_points[5]].y * frame_height]))
        v2 = np.linalg.norm(np.array([landmarks[eye_points[2]].x * frame_width,
                                      landmarks[eye_points[2]].y * frame_height])
                            - np.array([landmarks[eye_points[4]].x * frame_width,
                                        landmarks[eye_points[4]].y * frame_height]))
        h = np.linalg.norm(np.array([landmarks[eye_points[0]].x * frame_width,
                                     landmarks[eye_points[0]].y * frame_height])
                           - np.array([landmarks[eye_points[3]].x * frame_width,
                                       landmarks[eye_points[3]].y * frame_height]))
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def create_virtual_keyboard():
        nonlocal keyboard_window, keyboard_buttons, typed_text, keyboard_visible, caps_lock
        if keyboard_window:
            keyboard_window.destroy()

        keyboard_window = Tk()
        keyboard_window.title("Virtual Keyboard")
        keyboard_window.geometry("900x350+600+600")
        keyboard_window.attributes('-topmost', True)

        try:
            icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
            icon_image = icon_image.resize((128, 128), Image.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            keyboard_window.iconphoto(True, icon_photo)
        except Exception as e:
            output_queue.put(f"Error loading icon: {e}\n")

        def on_closing():
            nonlocal keyboard_visible, keyboard_window
            keyboard_visible = False
            keyboard_window.destroy()
            keyboard_window = None

        keyboard_window.protocol("WM_DELETE_WINDOW", on_closing)

        text_display = tk.Label(keyboard_window, text=typed_text, font=("Arial", 12))
        text_display.grid(row=0, column=0, columnspan=10, pady=5)

        def update_text(char):
            nonlocal typed_text, focused_input, caps_lock
            if char == 'capslock':
                caps_lock = not caps_lock
                update_button_labels()
                return

            if focused_input:
                pyautogui.click(focused_input[0], focused_input[1])
                time.sleep(0.1)
                if char == 'space':
                    pyautogui.press('space')
                    typed_text += ' '
                elif char == 'enter':
                    pyautogui.press('enter')
                    typed_text = ""
                elif char == 'delete':
                    pyautogui.press('backspace')
                    typed_text = typed_text[:-1] if typed_text else typed_text
                else:
                    char_to_write = char.upper() if caps_lock else char
                    pyautogui.write(char_to_write)
                    typed_text += char_to_write
            else:
                if char == 'space':
                    typed_text += ' '
                elif char == 'delete':
                    typed_text = typed_text[:-1] if typed_text else typed_text
                elif char == 'enter':
                    typed_text = ""
                else:
                    char_to_write = char.upper() if caps_lock else char
                    typed_text += char_to_write
            text_display.config(text=typed_text)

        def update_button_labels():
            for button, char in keyboard_buttons:
                if char in 'abcdefghijklmnopqrstuvwxyz':
                    button.config(text=char.upper() if caps_lock else char)
                elif char == 'capslock':
                    button.config(bg='lightblue' if caps_lock else 'SystemButtonFace')

        keyboard_buttons = []
        for row_idx, row in enumerate(keyboard_layout):
            for col_idx, char in enumerate(row):
                display_text = char.upper() if (
                                                       char in 'abcdefghijklmnopqrstuvwxyz' and caps_lock) or char == 'capslock' else char
                button = tk.Button(keyboard_window, text=display_text, width=8, height=2,
                                   command=lambda c=char: update_text(c),
                                   bg='lightblue' if char == 'capslock' and caps_lock else 'SystemButtonFace')
                button.grid(row=row_idx + 1, column=col_idx, padx=5, pady=2)
                keyboard_buttons.append((button, char))

        keyboard_window.update()

    def calibrate():
        calibration_points = []
        samples_per_point = 30
        center_samples = 50
        calibration_positions = [
            "Top Right", "Top Left", "Bottom Right", "Bottom Left",
            "Upper Center", "Lower Center", "Right Center", "Left Center", "Center"
        ]
        position_mappings = {
            "Top Right": (0.9, 0.1), "Top Left": (0.1, 0.1),
            "Bottom Right": (0.9, 0.9), "Bottom Left": (0.1, 0.9),
            "Upper Center": (0.5, 0.1), "Lower Center": (0.5, 0.9),
            "Right Center": (0.9, 0.5), "Left Center": (0.1, 0.5),
            "Center": (0.5, 0.5)
        }

        for point in calibration_positions:
            output_queue.put(f"Please look at the {point} point and wait for 3 seconds.\n")
            point_samples = []
            sample_count = center_samples if point == "Center" else samples_per_point
            start_time = time.time()
            while time.time() - start_time < 3:
                success, frame = cam.read()
                if not success:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        eye_center_x = (np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS]) +
                                        np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])) / 2
                        eye_center_y = (np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS]) +
                                        np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])) / 2
                        point_samples.append((eye_center_x, eye_center_y))
                        if len(point_samples) >= sample_count:
                            break
                cv2.waitKey(1)
            if point_samples:
                avg_x = np.mean([x for x, y in point_samples])
                avg_y = np.mean([y for x, y in point_samples])
                std_x = np.std([x for x, y in point_samples])
                std_y = np.std([y for x, y in point_samples])
                calibration_points.append((avg_x, avg_y, std_x, std_y))
                output_queue.put(
                    f"Eye center for {point}: ({avg_x:.3f}, {avg_y:.3f}), Standard deviation: x={std_x:.3f}, y={std_y:.3f}\n")
            else:
                output_queue.put(f"No face detected for {point}!\n")

        if len(calibration_points) < len(calibration_positions):
            output_queue.put("Calibration could not be completed, using default values.\n")
            return 0.25, 0.75, 0.25, 0.75, 0, 0

        x_coords, y_coords, x_stds, y_stds = zip(*calibration_points)
        x_min, x_max = min(x_coords) - max(x_stds), max(x_coords) + max(x_stds)
        y_min, y_max = min(y_coords) - max(y_stds), max(y_coords) + max(y_stds)

        center_idx = calibration_positions.index("Center")
        center_x, center_y = calibration_points[center_idx][:2]
        expected_center_x = (x_min + x_max) / 2
        expected_center_y = (y_min + y_max) / 2
        x_offset = expected_center_x - center_x
        y_offset = expected_center_y - center_y

        output_queue.put(
            f"Calibration results: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}\n")
        output_queue.put(f"Center correction: x_offset={x_offset:.3f}, y_offset={y_offset:.3f}\n")

        return x_min, x_max, y_min, y_max, x_offset, y_offset

    output_queue.put("Starting calibration...\n")
    x_min, x_max, y_min, y_max, x_offset, y_offset = calibrate()
    output_queue.put(
        f"Calibration completed: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}\n")

    pyautogui.moveTo(screen_width / 2, screen_height / 2)

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            output_queue.put("Could not capture camera image!\n")
            break

        display_width, display_height = 640, 480
        frame = cv2.resize(frame, (display_width, display_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
                left_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS])
                right_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])
                right_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])

                eye_center_x = (left_iris_center_x + right_iris_center_x) / 2
                eye_center_y = (left_iris_center_y + right_iris_center_y) / 2

                eye_center_x += x_offset
                eye_center_y += y_offset

                mouse_x = np.interp(eye_center_x, [x_min, x_max], [screen_width, 0])
                mouse_y = np.interp(eye_center_y, [y_min, y_max], [0, screen_height])

                mouse_x_history.append(mouse_x)
                mouse_y_history.append(mouse_y)
                mouse_x = np.mean(mouse_x_history)
                mouse_y = np.mean(mouse_y_history)

                mouse_x = smooth_factor * mouse_x + (1 - smooth_factor) * prev_mouse_x
                mouse_y = smooth_factor * mouse_y + (1 - smooth_factor) * prev_mouse_y

                if abs(mouse_x - prev_mouse_x) < MOVEMENT_THRESHOLD and abs(
                        mouse_y - prev_mouse_y) < MOVEMENT_THRESHOLD:
                    mouse_x, mouse_y = prev_mouse_x, prev_mouse_y

                if click_stabilization_cooldown == 0:
                    pyautogui.moveTo(mouse_x, mouse_y)

                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark, display_width, display_height)
                right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark, display_width, display_height)

                current_time = time.time()

                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    if both_eyes_closed_start is None:
                        both_eyes_closed_start = current_time
                        output_queue.put("Both eyes closed detected, timer started.\n")
                    elif current_time - both_eyes_closed_start >= BOTH_EYES_CLOSED_DURATION:
                        output_queue.put(
                            f"Both eyes closed for {BOTH_EYES_CLOSED_DURATION} seconds, application is closing.\n")
                        if keyboard_window:
                            keyboard_window.destroy()
                        cam.release()
                        cv2.destroyAllWindows()
                        face_mesh.close()
                        return
                else:
                    if both_eyes_closed_start is not None:
                        output_queue.put("Eyes opened, timer reset.\n")
                    both_eyes_closed_start = None

                if scroll_cooldown > 0:
                    scroll_cooldown -= 1
                    continue

                if left_ear < EAR_THRESHOLD:
                    left_blink_counter += 1
                    if left_blink_counter >= BLINK_DURATION and not left_click_active:
                        if not blink_sequence or (current_time - last_blink_time) > SCROLL_THRESHOLD:
                            blink_sequence.clear()
                        blink_sequence.append('left')
                        last_blink_time = current_time
                        pending_click = 'left'
                        left_click_active = True
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        output_queue.put("Left eye blink detected.\n")
                        if current_time - last_double_click_time <= double_click_window:
                            click_count += 1
                            if click_count >= 2:
                                focused_input = (mouse_x, mouse_y)
                                keyboard_visible = True
                                output_queue.put("Double click detected, opening virtual keyboard.\n")
                                create_virtual_keyboard()
                                click_count = 0
                                click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        else:
                            click_count = 1
                        last_double_click_time = current_time
                else:
                    left_blink_counter = 0
                    left_click_active = False

                if right_ear < EAR_THRESHOLD:
                    right_blink_counter += 1
                    if right_blink_counter >= BLINK_DURATION and not right_click_active:
                        if not blink_sequence or (current_time - last_blink_time) > SCROLL_THRESHOLD:
                            blink_sequence.clear()
                        blink_sequence.append('right')
                        last_blink_time = current_time
                        pending_click = 'right'
                        right_click_active = True
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        output_queue.put("Right eye blink detected.\n")
                else:
                    right_blink_counter = 0
                    right_click_active = False

                if len(blink_sequence) >= 2:
                    if blink_sequence[-2:] == ['right', 'left']:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        output_queue.put("Scrolled up.\n")
                        blink_sequence.clear()
                        pending_click = None
                        scroll_cooldown = 5
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    elif blink_sequence[-2:] == ['left', 'right']:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        output_queue.put("Scrolled down.\n")
                        blink_sequence.clear()
                        pending_click = None
                        scroll_cooldown = 5
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION

                if pending_click and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                    if pending_click == 'left':
                        pyautogui.click(button='left')
                        output_queue.put("Left click performed.\n")
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    elif pending_click == 'right':
                        pyautogui.click(button='right')
                        output_queue.put("Right click performed.\n")
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    blink_sequence.clear()
                    pending_click = None
                    scroll_cooldown = 5

                if blink_sequence and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                    blink_sequence.clear()
                    pending_click = None

                if click_stabilization_cooldown > 0:
                    click_stabilization_cooldown -= 1

                h, w, _ = frame.shape
                left_iris_x = int(left_iris_center_x * w)
                left_iris_y = int(left_iris_center_y * h)
                right_iris_x = int(right_iris_center_x * w)
                right_iris_y = int(right_iris_center_y * h)

                if left_ear >= EAR_THRESHOLD:
                    cv2.circle(frame, (left_iris_x, left_iris_y), 5, (0, 255, 0), -1)
                if right_ear >= EAR_THRESHOLD:
                    cv2.circle(frame, (right_iris_x, right_iris_y), 5, (0, 255, 0), -1)

        else:
            output_queue.put("No face detected!\n")
            both_eyes_closed_start = None

        cv2.imshow('Eye Mouse', frame)
        set_window_topmost('Eye Mouse')

        if keyboard_window:
            try:
                keyboard_window.update()
            except tk.TclError:
                keyboard_visible = False
                keyboard_window = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if keyboard_window:
        keyboard_window.destroy()


# HeadMouse fonksiyonu
def run_head_mouse():
    output_queue.put("Starting HeadMouse...\n")

    # Mediapipe yüz mesh modelini başlatma
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

    # Ekran boyutlarını alma
    screen_width, screen_height = pydirectinput.size()

    # Web kamerasını başlatma
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Göz takibi için landmark indisleri
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    FACE_OVAL = [10, 338, 297, 332, 284, 251,
                 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377,
                 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127,
                 162, 21, 54, 103, 67, 109]

    # Yumuşatma ayarları
    smooth_factor = 0.15
    speed_factor = 1.5
    mouse_x_history = deque(maxlen=20)
    mouse_y_history = deque(maxlen=20)
    prev_mouse_x, prev_mouse_y = screen_width / 2, screen_height / 2
    MOVEMENT_THRESHOLD = 5
    last_valid_mouse_x, last_valid_mouse_y = prev_mouse_x, prev_mouse_y
    last_movement_direction = (0, 0)
    no_face_counter = 0
    NO_FACE_MAX_FRAMES = 10
    initial_stabilization = 30

    # Göz kırpma ayarları
    EAR_THRESHOLD = 0.27
    BLINK_DURATION = 3
    left_blink_counter = 0
    right_blink_counter = 0
    left_click_active = False
    right_click_active = False
    right_blink_timestamps = deque(maxlen=2)
    double_blink_time_window = 0.5
    last_left_blink_time = 0
    last_right_blink_time = 0
    last_double_click_time = 0
    double_click_window = 0.5
    click_count = 0
    right_blink_count = 0
    left_ear_history = deque(maxlen=10)
    right_ear_history = deque(maxlen=10)
    SCROLL_THRESHOLD = 0.7
    SCROLL_AMOUNT = 300
    last_blink_time = 0
    blink_sequence = []
    scroll_cooldown = 0
    pending_click = None
    click_stabilization_cooldown = 0
    CLICK_STABILIZATION_DURATION = 15
    WASD_COOLDOWN = 5
    wasd_cooldown = 0
    last_pressed_key = None

    # Sanal klavye ayarları
    keyboard_visible = False
    keyboard_window = None
    typed_text = ""
    caps_lock = False
    keyboard_buttons = []
    keyboard_layout = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
        ['space', 'enter', 'delete', 'capslock']
    ]
    speech_recognizer = sr.Recognizer()
    speech_visible = False
    speech_window = None
    speech_queue = queue.Queue()
    focused_input = None
    exit_window = None

    def set_window_topmost(window_name):
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    def calculate_ear(eye_points, landmarks, frame_width, frame_height):
        v1 = np.linalg.norm(np.array([landmarks[eye_points[1]].x * frame_width,
                                      landmarks[eye_points[1]].y * frame_height])
                            - np.array([landmarks[eye_points[5]].x * frame_width,
                                        landmarks[eye_points[5]].y * frame_height]))
        v2 = np.linalg.norm(np.array([landmarks[eye_points[2]].x * frame_width,
                                      landmarks[eye_points[2]].y * frame_height])
                            - np.array([landmarks[eye_points[4]].x * frame_width,
                                        landmarks[eye_points[4]].y * frame_height]))
        h = np.linalg.norm(np.array([landmarks[eye_points[0]].x * frame_width,
                                     landmarks[eye_points[0]].y * frame_height])
                           - np.array([landmarks[eye_points[3]].x * frame_width,
                                       landmarks[eye_points[3]].y * frame_height]))
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def create_exit_window():
        nonlocal exit_window
        if exit_window:
            exit_window.destroy()

        exit_window = Tk()
        exit_window.title("Exit Control")
        exit_window.geometry("200x50+190+430")
        exit_window.attributes('-topmost', True)

        try:
            icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
            icon_image = icon_image.resize((32, 32), Image.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            exit_window.iconphoto(True, icon_photo)
        except Exception as e:
            output_queue.put(f"Error loading icon: {e}\n")

        def on_exit():
            nonlocal exit_window, keyboard_window, speech_window
            output_queue.put("Exit button pressed, application closing.\n")
            if keyboard_window:
                keyboard_window.destroy()
            if speech_window:
                speech_window.destroy()
            if exit_window:
                exit_window.destroy()
            cam.release()
            cv2.destroyAllWindows()
            face_mesh.close()
            return

        exit_button = tk.Button(exit_window, text="Exit", width=10, height=2, command=on_exit, bg='red', fg='white')
        exit_button.pack(pady=5)
        exit_window.update()

    def create_virtual_keyboard():
        nonlocal keyboard_window, keyboard_buttons, typed_text, keyboard_visible, caps_lock
        if keyboard_window:
            keyboard_window.destroy()

        keyboard_window = Tk()
        keyboard_window.title("Virtual Keyboard")
        keyboard_window.geometry("900x350+300+400")
        keyboard_window.attributes('-topmost', True)
        typed_text = ""

        try:
            icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
            icon_image = icon_image.resize((128, 128), Image.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            keyboard_window.iconphoto(True, icon_photo)
        except Exception as e:
            output_queue.put(f"Error loading icon: {e}\n")

        def on_closing():
            nonlocal keyboard_visible, keyboard_window
            keyboard_visible = False
            keyboard_window.destroy()
            keyboard_window = None

        keyboard_window.protocol("WM_DELETE_WINDOW", on_closing)

        text_display = tk.Label(keyboard_window, text=typed_text, font=("Arial", 12))
        text_display.grid(row=0, column=0, columnspan=10, pady=5)

        def update_text(char):
            nonlocal typed_text, focused_input, caps_lock
            if char == 'capslock':
                caps_lock = not caps_lock
                update_button_labels()
                return

            if focused_input:
                pyautogui.click(focused_input[0], focused_input[1])
                time.sleep(0.1)
                if char == 'space':
                    pyautogui.press('space')
                    typed_text += ' '
                elif char == 'enter':
                    pyautogui.press('enter')
                    typed_text = ""
                elif char == 'delete':
                    pyautogui.press('backspace')
                    typed_text = typed_text[:-1] if typed_text else typed_text
                else:
                    char_to_write = char.upper() if caps_lock else char
                    pyautogui.write(char_to_write)
                    typed_text += char_to_write
            else:
                if char == 'space':
                    typed_text += ' '
                elif char == 'delete':
                    typed_text = typed_text[:-1] if typed_text else typed_text
                elif char == 'enter':
                    typed_text = ""
                else:
                    char_to_write = char.upper() if caps_lock else char
                    typed_text += char_to_write
            text_display.config(text=typed_text)

        def update_button_labels():
            for button, char in keyboard_buttons:
                if char in 'abcdefghijklmnopqrstuvwxyz':
                    button.config(text=char.upper() if caps_lock else char)
                elif char == 'capslock':
                    button.config(bg='lightblue' if caps_lock else 'SystemButtonFace')

        keyboard_buttons = []
        for row_idx, row in enumerate(keyboard_layout):
            for col_idx, char in enumerate(row):
                display_text = char.upper() if (
                                                       char in 'abcdefghijklmnopqrstuvwxyz' and caps_lock) or char == 'capslock' else char
                button = tk.Button(keyboard_window, text=display_text, width=8, height=2,
                                   command=lambda c=char: update_text(c),
                                   bg='lightblue' if char == 'capslock' and caps_lock else 'SystemButtonFace')
                button.grid(row=row_idx + 1, column=col_idx, padx=5, pady=2)
                keyboard_buttons.append((button, char))

        keyboard_window.update()

    def create_speech_window():
        nonlocal speech_window, speech_visible, typed_text, speech_queue
        if speech_window:
            speech_window.destroy()

        speech_window = Tk()
        speech_window.title("Voiced Command")
        speech_window.geometry("400x200+600+400")
        speech_window.attributes('-topmost', True)

        try:
            icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
            icon_image = icon_image.resize((128, 128), Image.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            speech_window.iconphoto(True, icon_photo)
        except Exception as e:
            output_queue.put(f"Error loading icon: {e}\n")

        def on_closing():
            nonlocal speech_visible, speech_window
            speech_visible = False
            speech_window.destroy()
            speech_window = None

        speech_window.protocol("WM_DELETE_WINDOW", on_closing)

        text_display = tk.Label(speech_window, text="Speak...", font=("Arial", 12))
        text_display.pack(pady=10)

        def update_text_display():
            try:
                while not speech_queue.empty():
                    text = speech_queue.get_nowait()
                    text_display.config(text=text)
                    speech_window.update()
            except tk.TclError:
                pass

        def start_listening():
            def listen_in_background():
                nonlocal typed_text, focused_input
                speech_queue.put("Listening...")
                try:
                    with sr.Microphone() as source:
                        speech_recognizer.adjust_for_ambient_noise(source)
                        audio = speech_recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        recognized_text = speech_recognizer.recognize_google(audio, language='en-US')
                        speech_queue.put(recognized_text)
                        typed_text += recognized_text + " "
                        if focused_input:
                            pyautogui.click(focused_input[0], focused_input[1])
                            time.sleep(0.1)
                            pyperclip.copy(recognized_text)
                            pyautogui.hotkey('ctrl', 'v')
                            typed_text += recognized_text + " "
                        else:
                            typed_text += recognized_text + " "
                except sr.WaitTimeoutError:
                    speech_queue.put("No sound detected, try again.")
                except sr.UnknownValueError:
                    speech_queue.put("Cannot understand, please speak again.")
                except sr.RequestError:
                    speech_queue.put("Error: Please check your internet connection.")
                except Exception as e:
                    speech_queue.put(f"Error: {str(e)}")

            threading.Thread(target=listen_in_background, daemon=True).start()

        listen_button = tk.Button(speech_window, text="Listen", width=10, height=2, command=start_listening)
        listen_button.pack(pady=10)

        speech_window.after(100, lambda: update_text_display_loop(update_text_display))

        speech_window.update()

    def update_text_display_loop(update_func):
        if speech_window:
            update_func()
            speech_window.after(100, lambda: update_text_display_loop(update_func))

    def calibrate():
        calibration_points = []
        samples_per_point = 30
        center_samples = 1000
        calibration_positions = [
            "Top Right", "Top Left", "Bottom Right", "Bottom Left",
            "Upper Center", "Lower Center", "Right Center", "Left Center", "Center"
        ]
        position_mappings = {
            "Top Right": (0.9, 0.1), "Top Left": (0.1, 0.1),
            "Bottom Right": (0.9, 0.9), "Bottom Left": (0.1, 0.9),
            "Upper Center": (0.5, 0.1), "Lower Center": (0.5, 0.9),
            "Right Center": (0.9, 0.5), "Left Center": (0.1, 0.5),
            "Center": (0.5, 0.5)
        }

        for point in calibration_positions:
            output_queue.put(f"Please position your head at the {point} point of the screen and wait for 3 seconds.\n")
            point_samples = []
            sample_count = center_samples if point == "Center" else samples_per_point
            start_time = time.time()
            while time.time() - start_time < 3:
                success, frame = cam.read()
                if not success:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        nose_tip_x = face_landmarks.landmark[1].x
                        nose_tip_y = face_landmarks.landmark[1].y
                        point_samples.append((nose_tip_x, nose_tip_y))
                        if len(point_samples) >= sample_count:
                            break
                cv2.waitKey(1)
            if point_samples:
                avg_x = np.mean([x for x, y in point_samples])
                avg_y = np.mean([y for x, y in point_samples])
                std_x = np.std([x for x, y in point_samples])
                std_y = np.std([y for x, y in point_samples])
                calibration_points.append((avg_x, avg_y, std_x, std_y))
                output_queue.put(
                    f"Nose tip for {point}: ({avg_x:.3f}, {avg_y:.3f}), Standard deviation: x={std_x:.3f}, y={std_y:.3f}\n")
            else:
                output_queue.put(f"No face detected for {point}!\n")
            time.sleep(1)

        if len(calibration_points) < len(calibration_positions):
            output_queue.put("Calibration could not be completed, default values are used.\n")
            return 0.1, 0.9, 0.1, 0.9, 0, 0

        x_coords, y_coords, x_stds, y_stds = zip(*calibration_points)
        margin = 0.1
        x_min = min(x_coords) - max(x_stds) - margin
        x_max = max(x_coords) + max(x_stds) + margin
        y_min = min(y_coords) - max(y_stds) - margin
        y_max = max(y_coords) + max(y_stds) + margin

        left_points = [calibration_points[i] for i in [1, 3, 7]]
        right_points = [calibration_points[i] for i in [0, 2, 6]]
        left_avg_x = np.mean([p[0] for p in left_points])
        right_avg_x = np.mean([p[0] for p in right_points])
        if left_avg_x > right_avg_x:
            output_queue.put("Left-right asymmetry detected and corrected.\n")
            x_min = min(left_avg_x, right_avg_x) - margin
            x_max = max(left_avg_x, right_avg_x) + margin

        center_idx = calibration_positions.index("Center")
        center_x, center_y = calibration_points[center_idx][:2]
        expected_center_x, expected_center_y = 0.5, 0.5
        x_offset = expected_center_x - center_x
        y_offset = expected_center_y - center_y

        x_offset = max(min(x_offset, 0.05), -0.05)
        y_offset = max(min(y_offset, 0.05), -0.05)

        output_queue.put(
            f"Calibration Results: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}\n")
        output_queue.put(f"Center Correction: x_offset={x_offset:.3f}, y_offset={y_offset:.3f}\n")

        return x_min, x_max, y_min, y_max, x_offset, y_offset

    output_queue.put("Starting calibration...\n")
    x_min, x_max, y_min, y_max, x_offset, y_offset = calibrate()
    output_queue.put(
        f"Calibration completed: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}\n")

    pydirectinput.moveTo(screen_width // 2, screen_height // 2)
    prev_mouse_x, prev_mouse_y = screen_width // 2, screen_height // 2
    last_valid_mouse_x, last_valid_mouse_y = screen_width // 2, screen_height // 2
    last_movement_direction = (0, 0)
    mouse_x_history.clear()
    mouse_y_history.clear()
    mouse_x_history.append(screen_width // 2)
    mouse_y_history.append(screen_height // 2)

    def ensure_initial_center():
        output_queue.put("The mouse is aligned to the center of the screen.\n")
        pydirectinput.moveTo(screen_width // 2, screen_height // 2)

    create_exit_window()
    ensure_initial_center()

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            output_queue.put("Could not retrieve camera image!\n")
            continue

        display_width, display_height = 480, 360
        frame = cv2.resize(frame, (display_width, display_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            no_face_counter = 0
            for face_landmarks in results.multi_face_landmarks:
                nose_tip_x = face_landmarks.landmark[1].x
                nose_tip_y = face_landmarks.landmark[1].y

                nose_tip_x += x_offset
                nose_tip_y += y_offset

                nose_tip_x = max(x_min, min(nose_tip_x, x_max))
                nose_tip_y = max(y_min, min(nose_tip_y, y_max))

                mouse_x = np.interp(nose_tip_x, [x_min, x_max], [screen_width, 0]) * speed_factor
                mouse_y = np.interp(nose_tip_y, [y_min, y_max], [0, screen_height]) * speed_factor

                mouse_x = max(0, min(mouse_x, screen_width))
                mouse_y = max(0, min(mouse_y, screen_height))

                mouse_x_history.append(mouse_x)
                mouse_y_history.append(mouse_y)
                mouse_x = np.mean(mouse_x_history)
                mouse_y = np.mean(mouse_y_history)

                mouse_x = smooth_factor * mouse_x + (1 - smooth_factor) * prev_mouse_x
                mouse_y = smooth_factor * mouse_y + (1 - smooth_factor) * prev_mouse_y

                if abs(mouse_x - prev_mouse_x) < MOVEMENT_THRESHOLD and abs(
                        mouse_y - prev_mouse_y) < MOVEMENT_THRESHOLD:
                    mouse_x, mouse_y = prev_mouse_x, prev_mouse_y
                else:
                    last_movement_direction = (mouse_x - prev_mouse_x, mouse_y - prev_mouse_y)

                if initial_stabilization > 0:
                    mouse_x, mouse_y = screen_width // 2, screen_height // 2
                    pydirectinput.moveTo(int(mouse_x), int(mouse_y))
                    initial_stabilization -= 1
                elif click_stabilization_cooldown == 0:
                    pydirectinput.moveTo(int(mouse_x), int(mouse_y))

                last_valid_mouse_x, last_valid_mouse_y = mouse_x, mouse_y
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark, display_width, display_height)
                right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark, display_width, display_height)

                left_ear_history.append(left_ear)
                right_ear_history.append(right_ear)
                smoothed_left_ear = np.mean(left_ear_history)
                smoothed_right_ear = np.mean(right_ear_history)

                left_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
                left_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS])
                right_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])
                right_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])

                current_time = time.time()

                if smoothed_left_ear < EAR_THRESHOLD and smoothed_right_ear < EAR_THRESHOLD:
                    if wasd_cooldown == 0:
                        if last_pressed_key:
                            pydirectinput.keyUp(last_pressed_key)
                            output_queue.put(f"{last_pressed_key.upper()} key released.\n")
                            last_pressed_key = None
                        x_normalized = mouse_x / screen_width
                        y_normalized = mouse_y / screen_height
                        if y_normalized < 0.3 and not focused_input:
                            pydirectinput.keyDown('w')
                            last_pressed_key = 'w'
                            output_queue.put("W key pressed.\n")
                            wasd_cooldown = WASD_COOLDOWN
                        elif y_normalized > 0.7 and not focused_input:
                            pydirectinput.keyDown('s')
                            last_pressed_key = 's'
                            output_queue.put("S key pressed.\n")
                            wasd_cooldown = WASD_COOLDOWN
                        elif x_normalized < 0.3 and not focused_input:
                            pydirectinput.keyDown('a')
                            last_pressed_key = 'a'
                            output_queue.put("A key pressed.\n")
                            wasd_cooldown = WASD_COOLDOWN
                        elif x_normalized > 0.7 and not focused_input:
                            pydirectinput.keyDown('d')
                            last_pressed_key = 'd'
                            output_queue.put("D key pressed.\n")
                            wasd_cooldown = WASD_COOLDOWN
                        elif not focused_input:
                            pydirectinput.keyDown('space')
                            last_pressed_key = 'space'
                            output_queue.put("Space key pressed.\n")
                            wasd_cooldown = WASD_COOLDOWN
                else:
                    if last_pressed_key:
                        pydirectinput.keyUp(last_pressed_key)
                        output_queue.put(f"{last_pressed_key.upper()} key released.\n")
                        last_pressed_key = None

                if scroll_cooldown > 0:
                    scroll_cooldown -= 1
                    continue

                if wasd_cooldown > 0:
                    wasd_cooldown -= 1
                    continue

                is_head_stable = (abs(mouse_x - prev_mouse_x) < MOVEMENT_THRESHOLD and
                                  abs(mouse_y - prev_mouse_y) < MOVEMENT_THRESHOLD)

                if smoothed_left_ear < EAR_THRESHOLD:
                    left_blink_counter += 1
                    if left_blink_counter >= BLINK_DURATION and not left_click_active:
                        if not blink_sequence or (current_time - last_blink_time) > SCROLL_THRESHOLD:
                            blink_sequence.clear()
                        blink_sequence.append('left')
                        last_blink_time = current_time
                        last_left_blink_time = current_time
                        pending_click = 'left'
                        left_click_active = True
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        output_queue.put("Left eye blink detected.\n")
                        if current_time - last_double_click_time <= double_click_window:
                            click_count += 1
                            if click_count >= 2:
                                focused_input = (mouse_x, mouse_y)
                                keyboard_visible = True
                                output_queue.put("Double left eye blink detected, opening virtual keyboard.\n")
                                create_virtual_keyboard()
                                click_count = 0
                                click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        else:
                            click_count = 1
                        last_double_click_time = current_time
                else:
                    left_blink_counter = 0
                    left_click_active = False

                if smoothed_right_ear < EAR_THRESHOLD and is_head_stable:
                    right_blink_counter += 1
                    if right_blink_counter >= BLINK_DURATION and not right_click_active:
                        right_blink_timestamps.append(current_time)
                        right_blink_count += 1
                        last_blink_time = current_time
                        last_right_blink_time = current_time
                        right_click_active = True
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                        output_queue.put("Right eye blink detected.\n")
                        if len(right_blink_timestamps) >= 2:
                            if right_blink_timestamps[-1] - right_blink_timestamps[-2] <= double_click_window:
                                focused_input = (mouse_x, mouse_y)
                                speech_visible = True
                                output_queue.put("Double right eye blink detected, opening voice command window.\n")
                                create_speech_window()
                                right_blink_timestamps.clear()
                                right_blink_count = 0
                                blink_sequence.clear()
                                pending_click = None
                                click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                                continue
                        else:
                            if not blink_sequence or (current_time - last_blink_time) > SCROLL_THRESHOLD:
                                blink_sequence.clear()
                            blink_sequence.append('right')
                            pending_click = 'right'
                else:
                    right_blink_counter = 0
                    right_click_active = False

                if len(blink_sequence) >= 2:
                    if blink_sequence[-2:] == ['right', 'left']:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        output_queue.put("Scrolled up.\n")
                        blink_sequence.clear()
                        pending_click = None
                        scroll_cooldown = 5
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    elif blink_sequence[-2:] == ['left', 'right']:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        output_queue.put("Scrolled down.\n")
                        blink_sequence.clear()
                        pending_click = None
                        scroll_cooldown = 5
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION

                if pending_click and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                    if pending_click == 'left':
                        pydirectinput.click(button='left')
                        output_queue.put("Left click performed.\n")
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    elif pending_click == 'right' and len(right_blink_timestamps) < 2:
                        pydirectinput.click(button='right')
                        output_queue.put("Right click performed.\n")
                        click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    blink_sequence.clear()
                    pending_click = None
                    scroll_cooldown = 5

                if blink_sequence and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                    blink_sequence.clear()
                    pending_click = None

                if click_stabilization_cooldown > 0:
                    click_stabilization_cooldown -= 1

                h, w, _ = frame.shape
                left_iris_x = int(left_iris_center_x * w)
                left_iris_y = int(left_iris_center_y * h)
                right_iris_x = int(right_iris_center_x * w)
                right_iris_y = int(right_iris_center_y * h)

                if smoothed_left_ear >= EAR_THRESHOLD:
                    cv2.circle(frame, (left_iris_x, left_iris_y), 5, (0, 255, 0), -1)
                if smoothed_right_ear >= EAR_THRESHOLD:
                    cv2.circle(frame, (right_iris_x, right_iris_y), 5, (0, 255, 0), -1)

                points = []
                for idx in FACE_OVAL:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    points.append((x, y))
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
                cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)

        else:
            output_queue.put(f"No face detected, using last known movement direction: {last_movement_direction}\n")
            no_face_counter += 1
            if no_face_counter <= NO_FACE_MAX_FRAMES:
                mouse_x = last_valid_mouse_x + last_movement_direction[0] * speed_factor
                mouse_y = last_valid_mouse_y + last_movement_direction[1] * speed_factor
                mouse_x = max(0, min(mouse_x, screen_width))
                mouse_y = max(0, min(mouse_y, screen_height))
                if click_stabilization_cooldown == 0 and initial_stabilization == 0:
                    pydirectinput.moveTo(int(mouse_x), int(mouse_y))
                    output_queue.put(f"No face detected, mouse moved to: ({mouse_x:.2f}, {mouse_y:.2f})\n")
                last_valid_mouse_x, last_valid_mouse_y = mouse_x, mouse_y
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
            else:
                output_queue.put("No face detected, mouse position preserved.\n")
                last_movement_direction = (0, 0)

        cv2.imshow('Head Mouse', frame)
        set_window_topmost('Head Mouse')

        if keyboard_window:
            try:
                keyboard_window.update()
            except tk.TclError:
                keyboard_visible = False
                keyboard_window = None
        if speech_window:
            try:
                speech_window.update()
            except tk.TclError:
                speech_visible = False
                speech_window = None
        if exit_window:
            try:
                exit_window.update()
            except tk.TclError:
                exit_window = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if last_pressed_key:
                pydirectinput.keyUp(last_pressed_key)
                output_queue.put(f"{last_pressed_key.upper()} key released.\n")
            break

    cam.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if keyboard_window:
        keyboard_window.destroy()
    if speech_window:
        speech_window.destroy()
    if exit_window:
        exit_window.destroy()


# Ana pencereyi oluşturma
root = Tk()
root.title("ANTI HAND Digital Mouse")
root.geometry("600x700")
root.resizable(width=False, height=False)

# Uygulama logosunu ekleme
try:
    icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
    icon_image = icon_image.resize((128, 128), Image.LANCZOS)
    icon_photo = ImageTk.PhotoImage(icon_image)
    root.iconphoto(True, icon_photo)
except Exception as e:
    output_queue.put(f"Error: Failed to load icon: {e}\n")

# Ekran boyutlarını alma ve pencereyi konumlandırma
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 600
window_height = 700
x = (screen_width // 2) - (window_width // 2)
y = 50
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Sayfaları yönetmek için bir konteyner çerçeve
container = Frame(root)
container.pack(fill="both", expand=True)

# Sayfaları saklamak için bir sözlük
pages = {}
current_page = 0

# Birinci sayfa: Eye Mouse Kullanıcı Kılavuzu
page1 = Frame(container)
pages[0] = page1
page1_text = Text(page1, height=30, width=60, font=("Arial", 10), wrap="word")
page1_text.pack(pady=20)
page1_text.insert(END, """User Manual

Informations for Eye Mouse:
1. When the application starts, it will calibrate your eye locations.
Please look at the following computer screen corners for 3 seconds in this order:
Top Right, Top Left, Bottom Right, Bottom Left, Upper Center,
Lower Center, Right Center, Left Center, and Center of the screen.
Now the mouse will follow your eyes.

2. Mouse button controls:
- For 'Right Click', blink your 'Right Eye'.
- For 'Left Click', blink your 'Left Eye'.
- For 'Scroll Down', blink your 'Left Eye' then 'Right Eye' repeatedly.
- For 'Scroll Up', blink your 'Right Eye' then 'Left Eye' repeatedly.
- To open the 'Virtual Keyboard', blink your 'Left Eye' twice in a row.
- To close the application, keep 'Both Eyes' closed for 2 seconds.""")
page1_text.config(state="disabled")

# İkinci sayfa: Head Mouse Kullanıcı Kılavuzu
page2 = Frame(container)
pages[1] = page2
page2_text = Text(page2, height=30, width=60, font=("Arial", 10), wrap="word")
page2_text.pack(pady=20)
page2_text.insert(END, """Informations for Head Mouse:
1. When the application starts, it will calibrate your nose location.
Please position your head at the following computer screen corners for 3 seconds in this order:
Top Right, Top Left, Bottom Right, Bottom Left, Upper Center,
Lower Center, Right Center, Left Center, and Center of the screen.
Now the mouse will follow your nose.

2. Mouse button controls:
- For 'Right Click', blink your 'Right Eye'.
- For 'Left Click', blink your 'Left Eye'.
- For 'Scroll Down', blink your 'Left Eye' then 'Right Eye' repeatedly.
- For 'Scroll Up', blink your 'Right Eye' then 'Left Eye' repeatedly.
- To open the 'Virtual Keyboard', blink your 'Left Eye' twice in a row.
- To open the 'Voice Command' window, blink your 'Right Eye' twice in a row.
- To close the application, click the 'Exit Button' under the application camera.
- In Computer Games:
To press the 'W Button', blink 'Both Eyes' while positioning your head at the 'Upper' part of the screen.
To press the 'A Button', blink 'Both Eyes' while positioning your head at the 'Left Center' of the screen.
To press the 'S Button', blink 'Both Eyes' while positioning your head at the 'Lower' part of the screen.
To press the 'D Button', blink 'Both Eyes' while positioning your head at the 'Right Center' of the screen.
To press the 'Space Button', blink 'Both Eyes' while positioning your head at the 'Center' of the screen.""")
page2_text.config(state="disabled")

# Üçüncü sayfa: Mevcut uygulama arayüzü
page3 = Frame(container)
pages[2] = page3
label = Label(page3, text="Which application do you choose?", font=("Arial", 14))
label.pack(pady=20)
eye_button = Button(page3, text="Eye Mouse", font=("Arial", 12), width=15, height=2,
                    command=lambda: threading.Thread(target=run_eye_mouse, daemon=True).start())
eye_button.pack(pady=10)
head_button = Button(page3, text="Head Mouse", font=("Arial", 12), width=15, height=2,
                     command=lambda: threading.Thread(target=run_head_mouse, daemon=True).start())
head_button.pack(pady=10)
output_label = Label(page3, text="Output Log:", font=("Arial", 12))
output_label.pack(pady=10)
output_text = Text(page3, height=20, width=60, font=("Arial", 10))
output_text.pack(pady=10)
scrollbar = Scrollbar(page3, orient="vertical", command=output_text.yview)
scrollbar.pack(side="right", fill="y")
output_text.config(yscrollcommand=scrollbar.set)


# Sayfaları değiştirme fonksiyonları
def show_page(page_number):
    global current_page
    current_page = page_number
    for i, page in pages.items():
        if i == page_number:
            page.pack(fill="both", expand=True)
        else:
            page.pack_forget()
    back_button.config(state="normal" if current_page > 0 else "disabled")
    next_button.config(state="normal" if current_page < 2 else "disabled")


def go_back():
    if current_page > 0:
        show_page(current_page - 1)


def go_forward():
    if current_page < 2:
        show_page(current_page + 1)


# İleri ve Geri düğmeleri
nav_frame = Frame(root)
nav_frame.pack(side="bottom", fill="x")
back_button = Button(nav_frame, text="Back", font=("Arial", 12), width=10, command=go_back)
back_button.pack(side="left", padx=10, pady=10)
next_button = Button(nav_frame, text="Next", font=("Arial", 12), width=10, command=go_forward)
next_button.pack(side="right", padx=10, pady=10)

# İlk sayfayı göster
show_page(0)

# Çıktı güncelleme döngüsünü başlat
root.after(100, update_output_text)

# Pencereyi çalıştırma
root.mainloop()