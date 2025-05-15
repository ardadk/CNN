import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model("C:/Users/Arda/Desktop/a/mymodel.keras")

# Etiketler
waste_labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
target_size = (224, 224)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    # Kameradan görüntü oku
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # Görüntü üzerinde kayar pencere (sliding window) ile nesneleri tespit et
    frame_height, frame_width, _ = frame.shape
    detection_boxes = []
    window_size = 100  # Pencere boyutu
    stride = 50        # Kaydırma adımı

    for y in range(0, frame_height - window_size, stride):
        for x in range(0, frame_width - window_size, stride):
            # Pencereyi kes
            window = frame[y:y+window_size, x:x+window_size]
            resized_window = cv2.resize(window, target_size)  # Modele uygun boyut
            img_array = np.expand_dims(resized_window, axis=0) / 255.0  # Normalizasyon

            # Tahmin yap
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]

            # Güven eşiği üzerinde ise algılamayı kaydet
            if confidence > 0.7:
                detection_boxes.append((x, y, x+window_size, y+window_size, predicted_class, confidence))

    # Tespit edilen nesneleri çerçevele
    for box in detection_boxes:
        x1, y1, x2, y2, predicted_class, confidence = box
        label = f"{waste_labels[predicted_class]} ({confidence*100:.2f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Çerçeve
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Etiket

    # Sonucu göster
    cv2.imshow("Garbage Object Detection", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
