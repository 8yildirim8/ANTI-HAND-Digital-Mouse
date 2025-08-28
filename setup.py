from cx_Freeze import setup, Executable
import sys

# Mediapipe ve diğer bağımlılıkların veri dosyalarını ekle
build_exe_options = {
    "packages": ["mediapipe", "cv2", "speech_recognition", "pygame", "matplotlib"],
    "includes": ["mediapipe.python.solutions", "matplotlib.pyplot"],
    "include_files": [
        (".venv1\\Lib\\site-packages\\mediapipe", "mediapipe"),
        (".venv1\\Lib\\site-packages\\cv2", "cv2"),
        (".venv1\\Lib\\site-packages\\speech_recognition", "speech_recognition"),
        (".venv1\\Lib\\site-packages\\mediapipe\\modules\\hand_landmark\\hand_landmark_full.tflite", "mediapipe/modules/hand_landmark/hand_landmark_full.tflite")
    ],
    "excludes": [],
}

# Windows için base ayarı
base = None
if sys.platform == "win32":
    base = "Console"

setup(
    name="ANTI_HAND_Digital_Mouse",
    version="0.1",
    description="Digital Mouse Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("ANTI_HAND_Digital_Mouse.py", base=base, icon=r"C:\Users\pc\EyeHead.ico")]
)

