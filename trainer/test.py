from pyzbar.pyzbar import decode
from PIL import Image

img = Image.open("zerodha_qr.png")
decoded = decode(img)
print(decoded[0].data.decode())
