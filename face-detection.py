import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('pas foto.jpg')

# Cek apakah gambar berhasil dimuat
if img is None:
    print("Error: Could not read the image. Check file path or integrity.")
    exit() # Keluar dari program jika gambar tidak dimuat

# Konversi gambar ke grayscale (umumnya lebih baik untuk deteksi wajah)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Gambar kotak di sekitar wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Tampilkan gambar dengan kotak wajah
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()