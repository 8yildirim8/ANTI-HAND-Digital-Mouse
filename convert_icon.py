from PIL import Image

# PNG dosyasını yükle
img = Image.open(r"C:\Users\pc\EyeHead.png")

# ICO dosyasına kaydet (örneğin, 256x256 boyutunda)
img.save(r"C:\Users\pc\EyeHead.ico", format="ICO", sizes=[(256, 256)])

