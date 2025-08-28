# Kullanılan kütüphaneler
import subprocess
import sys
import os
from tkinter import Tk, Label, Button, Text, Scrollbar, END, Frame
import threading
import queue
from PIL import Image, ImageTk

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

# Betik çalıştırma ve çıktıyı yakalama
def run_script(script_name):
    def stream_output(process):
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_queue.put(output.strip() + '\n')
        process.stdout.close()

    try:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        if not os.path.exists(script_path):
            output_queue.put(f"Error: {script_name} not found at {script_path}\n")
            return

        process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        threading.Thread(target=stream_output, args=(process,), daemon=True).start()

    except Exception as e:
        output_queue.put(f"Error: Cannot run {script_name}: {str(e)}\n")

# Uygulama versiyonlarını tuşlara tıklama sonrası çalıştırma
def run_eye_mouse():
    run_script("EyeMouse.py")

def run_head_mouse():
    run_script("HeadMouse.py")

# Ana pencereyi oluşturma
root = Tk()
root.title("Mouse Application")
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
1. When application starts it will calibrate your eye locations,
so you should look at these computer screen corners for 3 seconds 
by this sort:
Top Right, Top Left, Bottom Right, Bottom Left, Upper Center, 
Lower Center, Right Center, Left Center and Center of the screen.
Now mouse will follow your eyes.

2. Mouse button controls:
- For 'Right Click' blink your 'Right Eye'.
- For 'Left Click' blink your 'Left Eye'.
- For 'Scroll Down' blink your 'Left Eye' then 'Right Eye' repeatedly.
- For 'Scroll Up' blink your 'Right Eye' then 'Left Eye' repeatedly.
- For open the 'Virtual Keyboard' blink your 'Left Eye' twice in a row.
- For 'Closing the Application' keep your 'Both Eyes' closed for 2 seconds.""")
page1_text.config(state="disabled")

# İkinci sayfa: Head Mouse Kullanıcı Kılavuzu
page2 = Frame(container)
pages[1] = page2
page2_text = Text(page2, height=30, width=60, font=("Arial", 10), wrap="word")
page2_text.pack(pady=20)
page2_text.insert(END, """Informations for Head Mouse:
1. When application starts it will calibrate your nose location,
so you should position your head at these computer screen corners for 3 
seconds by this sort:
Top Right, Top Left, Bottom Right, Bottom Left, Upper Center, 
Lower Center, Right Center, Left Center and Center of the screen.
Now mouse will follow your nose.

2. Mouse button controls:
- For 'Right Click' blink your 'Right Eye'.
- For 'Left Click' blink your 'Left Eye'.
- For 'Scroll Down' blink your 'Left Eye' then 'Right Eye' repeatedly.
- For 'Scroll Up' blink your 'Right Eye' then 'Left Eye' repeatedly.
- For open the 'Virtual Keyboard' blink your 'Left Eye' twice in a row.
- For open the 'Voiced Command' blink your 'Right Eye' twice in a row.
- For 'Closing the Application' you can click the 'Exit Button' under the application camera.
- In Computer Games:
For click 'W Button' blink your 'Both Eyes' while positon your head 'Upper of the Screen'.
For click 'A Button' blink your 'Both Eyes' while positon your head 'Left Center of the Screen'.
For click 'S Button' blink your 'Both Eyes' while positon your head 'Lower of the Screen'.
For click 'D Button' blink your 'Both Eyes' while positon your head 'Right Center of the Screen'.
For click 'Space Button' blink your 'Both Eyes' while positon your head 'Center of the Screen'.""")
page2_text.config(state="disabled")

# Üçüncü sayfa: Mevcut uygulama arayüzü
page3 = Frame(container)
pages[2] = page3
label = Label(page3, text="Which application do you choose?", font=("Arial", 14))
label.pack(pady=20)
eye_button = Button(page3, text="Eye Mouse", font=("Arial", 12), width=15, height=2, command=run_eye_mouse)
eye_button.pack(pady=10)
head_button = Button(page3, text="Head Mouse", font=("Arial", 12), width=15, height=2, command=run_head_mouse)
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
        if i == current_page:
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