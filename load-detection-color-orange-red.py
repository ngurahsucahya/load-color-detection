import cv2
import numpy as np
import math



# Fungsi untuk mendeteksi satu objek oranye dalam frame
def detect_orange_object(frame):
    # Konversi frame ke dalam ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tentukan range warna oranye dalam HSV
    lower_orange = np.array([0, 50, 50])
    upper_orange = np.array([30, 255, 255])

    # Buat mask dengan menggunakan range warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Gunakan operasi morfologi untuk membersihkan noise dalam mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Temukan kontur objek oranye dalam mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi pusat objek
    object_center = None

    # Gambar kotak batas dan lingkaran pusat untuk objek oranye yang terdeteksi
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter objek berdasarkan ukuran area
            x, y, w, h = cv2.boundingRect(contour)

            # Hitung titik tengah objek
            center_x = x + w // 2
            center_y = y + h // 2
            object_center = (center_x, center_y)

            # Hitung titik tengah layar
            screen_center_x = frame.shape[1] // 2
            screen_center_y = frame.shape[0] // 2

            # Mendapatkan kuadran objek oranye
            quadrant = get_quadrant(center_x, center_y, frame.shape[1], frame.shape[0], object_center)
            quadrant_text = f"Quadrant: {quadrant}"

            # Tambahkan teks kuadran di atas kotak
            cv2.putText(frame, quadrant_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Gambar kotak batas
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tambahkan teks "Orange" di atas kotak
            cv2.putText(frame, "Orange", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Gambar lingkaran pusat objek
            cv2.circle(frame, object_center, 5, (0, 0, 255), -1)

            # Gambar garis menghubungkan titik tengah layar dan titik tengah objek
            cv2.line(frame, (screen_center_x, screen_center_y), (center_x, center_y), (255, 0, 0), 2)

            break  # Hentikan setelah mendeteksi satu objek
    return frame, object_center


def detect_red_object(frame):
    # Konversi frame ke dalam ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tentukan range warna merah dalam HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([0, 255, 255])

    # Buat mask dengan menggunakan range warna merah
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Gunakan operasi morfologi untuk membersihkan noise dalam mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Temukan kontur objek merah dalam mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi pusat objek
    object_center = None

    # Gambar kotak batas dan lingkaran pusat untuk objek merah yang terdeteksi
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter objek berdasarkan ukuran area
            x, y, w, h = cv2.boundingRect(contour)

            # Hitung titik tengah objek
            center_x = x + w // 2
            center_y = y + h // 2
            object_center = (center_x, center_y)

            # Hitung titik tengah layar
            screen_center_x = frame.shape[1] // 2
            screen_center_y = frame.shape[0] // 2

            # Mendapatkan kuadran objek red
            quadrant = get_quadrant(center_x, center_y, frame.shape[1], frame.shape[0], object_center)
            quadrant_text = f"Quadrant: {quadrant}"

            # Tambahkan teks kuadran di atas kotak
            cv2.putText(frame, quadrant_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Gambar kotak batas
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tambahkan teks "Orange" di atas kotak
            cv2.putText(frame, "Red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Gambar lingkaran pusat objek
            cv2.circle(frame, object_center, 5, (0, 0, 255), -1)

            # Gambar garis menghubungkan titik tengah layar dan titik tengah objek
            cv2.line(frame, (screen_center_x, screen_center_y), (center_x, center_y), (255, 0, 0), 2)

            break  # Hentikan setelah mendeteksi satu objek

    return frame, object_center

# Determine the quadrant and calculate length
def get_quadrant(cx, cy, width, height, object_center):
    if cx < width / 2:
        if cy < height / 2:
            return "Top Left"
        else:
            return "Bottom Left"
    else:
        if cy < height / 2:
            return "Top Right"
        else:
            return "Bottom Right"

    if object_center is not None:
        quadrant = get_quadrant(object_center[0], object_center[1], frame.shape[1], frame.shape[0], object_center)
        quadrant_text = f"Quadrant: {quadrant}"
        cv2.putText(frame, quadrant_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        dpi = 96
        # Calculate length in centimeters
        pixel_length = math.sqrt((object_center[0] - screen_center_x) ** 2 + (object_center[1] - screen_center_y) ** 2)
        cm_per_inch = 2.54  # 1 inch = 2.54 cm
        conversion_factor = cm_per_inch / dpi  # Faktor konversi pixel ke sentimeter
        length_cm = pixel_length * conversion_factor

        # Add length text to the frame
        length_text = f"Length: {length_cm:.2f} cm"
        cv2.putText(frame, length_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, object_center

# Inisialisasi kamera atau baca video
cap = cv2.VideoCapture(3)  # Menggunakan kamera default (ganti dengan path file video jika perlu)

# Variabel untuk dpi dan cm_per_inch
dpi = 100
cm_per_inch = 2.54  # 1 inch = 2.54 cm

while True:
    ret, frame = cap.read()  # Baca frame dari video/kamera

    if not ret:
        break

    # Gambar lingkaran pusat layar
    screen_center_x = frame.shape[1] // 2
    screen_center_y = frame.shape[0] // 2
    cv2.circle(frame, (screen_center_x, screen_center_y), 10, (255, 255, 255), -1)

    # Deteksi satu objek oranye dalam frame
    result_frame, object_center_orange = detect_orange_object(frame)

    # Deteksi satu objek merah dalam frame
    result_frame, object_center_red = detect_red_object(result_frame)

    # Tampilkan koordinat pusat objek oranye jika terdeteksi
    if object_center_orange is not None:
        print(f'Pusat Objek Oranye: {object_center_orange}')

        # Inisialisasi variabel untuk jarak oranye
        distance_orange_cm = None

        # Hitung jarak oranye jika objek oranye terdeteksi
        if object_center_orange is not None:
            # Menggunakan formula jarak Euclidean untuk menghitung jarak antara objek oranye dan pusat layar
            distance_orange_cm = math.sqrt((object_center_orange[0] - screen_center_x) ** 2 + (
                    object_center_orange[1] - screen_center_y) ** 2) / dpi * cm_per_inch

            # Tampilkan jarak oranye dalam sentimeter di frame
            distance_orange_text = f"Orange Distance: {distance_orange_cm:.2f} cm"
            cv2.putText(result_frame, distance_orange_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)


    # Tampilkan koordinat pusat objek merah jika terdeteksi
    if object_center_red is not None:
        print(f'Pusat Objek Merah: {object_center_red}')

        # Inisialisasi variabel untuk jarak merah
        distance_red_cm = None

        # Hitung jarak merah jika objek merah terdeteksi
        if object_center_red is not None:
            # Menggunakan formula jarak Euclidean untuk menghitung jarak antara objek merah dan pusat layar
            distance_red_cm = math.sqrt((object_center_red[0] - screen_center_x) ** 2 + (
                    object_center_red[1] - screen_center_y) ** 2) / dpi * cm_per_inch

            # Tampilkan jarak merah dalam sentimeter di frame
            distance_red_text = f"Red Distance: {distance_red_cm:.2f} cm"
            cv2.putText(result_frame, distance_red_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame hasil deteksi
    cv2.imshow('Object Detection', result_frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera atau video
cap.release()
cv2.destroyAllWindows()