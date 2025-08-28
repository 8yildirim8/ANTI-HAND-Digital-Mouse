import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import Tk
import win32gui
import win32con
from PIL import Image, ImageTk

# PyAutoGUI ayarları
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

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

# Göz kırpma için landmark indisleri
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

# Scroll için ek ayarlar
SCROLL_THRESHOLD = 0.7
SCROLL_AMOUNT = 300
last_blink_time = 0
blink_sequence = []
scroll_cooldown = 0
pending_click = None

# Tıklama stabilizasyonu için ayarlar
click_stabilization_cooldown = 0
CLICK_STABILIZATION_DURATION = 15

# Çift göz kırpma ile kapanma için ayarlar
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

# Odaklanılan giriş alanı
focused_input = None

# OpenCV penceresini her zaman üstte tutma fonksiyonu
def set_window_topmost(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

# Göz açıklık oranını (EAR) hesaplama fonksiyonu
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

# Sanal klavye oluşturma fonksiyonu
def create_virtual_keyboard():
    global keyboard_window, keyboard_buttons, typed_text, keyboard_visible, caps_lock
    if keyboard_window:
        keyboard_window.destroy()

    keyboard_window = Tk()
    keyboard_window.title("Virtual Keyboard")
    keyboard_window.geometry("900x350+600+600")
    keyboard_window.attributes('-topmost', True)

    # Uyulama logosunu ekleme
    try:
        icon_image = Image.open(r"C:\Users\pc\EyeHead(2).png")
        icon_image = icon_image.resize((128, 128), Image.LANCZOS)
        icon_photo = ImageTk.PhotoImage(icon_image)
        keyboard_window.iconphoto(True, icon_photo)
    except Exception as e:
        print(f"Simge yüklenirken hata oluştu: {e}")

    def on_closing():
        global keyboard_visible, keyboard_window
        keyboard_visible = False
        keyboard_window.destroy()
        keyboard_window = None

    keyboard_window.protocol("WM_DELETE_WINDOW", on_closing)

    text_display = tk.Label(keyboard_window, text=typed_text, font=("Arial", 12))
    text_display.grid(row=0, column=0, columnspan=10, pady=5)

    def update_text(char):
        global typed_text, focused_input, caps_lock
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

# Kalibrasyon fonksiyonu
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
        print(f"Please look at the {point} point and wait for 3 seconds.")
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
            print(
                f"Eye center for {point}: ({avg_x:.3f}, {avg_y:.3f}), Standard deviation: x={std_x:.3f}, y={std_y:.3f}")
        else:
            print(f"No face detected for {point}!")

    if len(calibration_points) < len(calibration_positions):
        print("Calibration could not be completed, using default values.")
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

    print(f"Calibration results: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}")
    print(f"Center correction: x_offset={x_offset:.3f}, y_offset={y_offset:.3f}")

    return x_min, x_max, y_min, y_max, x_offset, y_offset

# Kalibrasyonu çalıştırma
print("Calibration is starting...")
x_min, x_max, y_min, y_max, x_offset, y_offset = calibrate()
print(f"Calibration completed: x_min={x_min:.3f}, x_max={x_max:.3f}, y_min={y_min:.3f}, y_max={y_max:.3f}")

# Fareyi ekranın ortasına konumlandır
pyautogui.moveTo(screen_width / 2, screen_height / 2)

# Uygulamayı çalıştırma
while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera image could not be captured!")
        break

    # Görüntüyü yeniden boyutlandırma
    display_width, display_height = 640, 480
    frame = cv2.resize(frame, (display_width, display_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Gözlerin iris merkezlerini hesaplama
            left_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
            left_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in LEFT_IRIS])
            right_iris_center_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])
            right_iris_center_y = np.mean([face_landmarks.landmark[i].y for i in RIGHT_IRIS])

            # Göz merkezi hesaplama
            eye_center_x = (left_iris_center_x + right_iris_center_x) / 2
            eye_center_y = (left_iris_center_y + right_iris_center_y) / 2

            # Düzeltme faktörünü uygulama
            eye_center_x += x_offset
            eye_center_y += y_offset

            # Haritalama
            mouse_x = np.interp(eye_center_x, [x_min, x_max], [screen_width, 0])
            mouse_y = np.interp(eye_center_y, [y_min, y_max], [0, screen_height])

            # Hareketli ortalama filtresi
            mouse_x_history.append(mouse_x)
            mouse_y_history.append(mouse_y)
            mouse_x = np.mean(mouse_x_history)
            mouse_y = np.mean(mouse_y_history)

            # Yumuşatma
            mouse_x = smooth_factor * mouse_x + (1 - smooth_factor) * prev_mouse_x
            mouse_y = smooth_factor * mouse_y + (1 - smooth_factor) * prev_mouse_y

            # Hareket eşiği kontrolü
            if abs(mouse_x - prev_mouse_x) < MOVEMENT_THRESHOLD and abs(mouse_y - prev_mouse_y) < MOVEMENT_THRESHOLD:
                mouse_x, mouse_y = prev_mouse_x, prev_mouse_y

            # Fare hareketini yalnızca stabilizasyon soğuma süresi sıfır olduğunda güncelle
            if click_stabilization_cooldown == 0:
                pyautogui.moveTo(mouse_x, mouse_y)

            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            # Göz kırpma algılama
            left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark, display_width, display_height)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark, display_width, display_height)

            current_time = time.time()

            # Çift göz kırpma ile kapanma kontrolü
            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                if both_eyes_closed_start is None:
                    both_eyes_closed_start = current_time
                    print("Both eyes closed detected, timer started.")
                elif current_time - both_eyes_closed_start >= BOTH_EYES_CLOSED_DURATION:
                    print(f"Both eyes closed for {BOTH_EYES_CLOSED_DURATION} seconds, application is closing.")
                    if keyboard_window:
                        keyboard_window.destroy()
                    cam.release()
                    cv2.destroyAllWindows()
                    face_mesh.close()
                    exit()
            else:
                if both_eyes_closed_start is not None:
                    print("Eyes opened, timer reset.")
                both_eyes_closed_start = None

            # Kaydırma ve tıklama soğuma süresi kontrolü
            if scroll_cooldown > 0:
                scroll_cooldown -= 1
                continue

            # Sol göz kırpma algılama
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
                    print("Left eye blink detected.")
                    # Çift tıklama kontrolü
                    if current_time - last_double_click_time <= double_click_window:
                        click_count += 1
                        if click_count >= 2:
                            focused_input = (mouse_x, mouse_y)
                            keyboard_visible = True
                            print("Double click detected, virtual keyboard is opening.")
                            create_virtual_keyboard()
                            click_count = 0
                            click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                    else:
                        click_count = 1
                    last_double_click_time = current_time
            else:
                left_blink_counter = 0
                left_click_active = False

            # Sağ göz kırpma algılama
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
                    print("Right eye blink detected.")
            else:
                right_blink_counter = 0
                right_click_active = False

            # Scroll için kırpma sırası kontrolü
            if len(blink_sequence) >= 2:
                if blink_sequence[-2:] == ['right', 'left']:
                    pyautogui.scroll(SCROLL_AMOUNT)
                    print("Scrolled up.")
                    blink_sequence.clear()
                    pending_click = None
                    scroll_cooldown = 5
                    click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                elif blink_sequence[-2:] == ['left', 'right']:
                    pyautogui.scroll(-SCROLL_AMOUNT)
                    print("Scrolled down.")
                    blink_sequence.clear()
                    pending_click = None
                    scroll_cooldown = 5
                    click_stabilization_cooldown = CLICK_STABILIZATION_DURATION

            # Tıklama için zaman aşımı kontrolü
            if pending_click and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                if pending_click == 'left':
                    pyautogui.click(button='left')
                    print("Left click performed.")
                    click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                elif pending_click == 'right':
                    pyautogui.click(button='right')
                    print("Right click performed.")
                    click_stabilization_cooldown = CLICK_STABILIZATION_DURATION
                blink_sequence.clear()
                pending_click = None
                scroll_cooldown = 5

            # Zaman aşımı: Eğer kırpma sırası tamamlanmazsa sıfırla
            if blink_sequence and (current_time - last_blink_time) > SCROLL_THRESHOLD:
                blink_sequence.clear()
                pending_click = None

            # Stabilizasyon soğuma süresini azalt
            if click_stabilization_cooldown > 0:
                click_stabilization_cooldown -= 1

            # Görselleştirme
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
        print("No face detected!")
        both_eyes_closed_start = None

    cv2.imshow('Eye Mouse', frame)

    # OpenCV penceresini her zaman üstte tutma
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