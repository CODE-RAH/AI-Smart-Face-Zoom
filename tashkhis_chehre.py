import cv2

# بارگذاری مدل تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# اتصال به وبکم با رزولوشن بالا
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تبدیل تصویر به خاکستری و حذف نویز برای تشخیص بهتر
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # حذف نویز تصویر
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # رسم کادر دور چهره
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # تنظیم زوم بر اساس اندازه چهره
        if w > 300 and h > 300:  # چهره نزدیک است -> زوم زیاد
            zoom_factor = 2.0
        elif w > 150 and h > 150:  # چهره متوسط است -> زوم متوسط
            zoom_factor = 1.5
        else:  # چهره دور است -> زوم کم
            zoom_factor = 1.2

        # برش و زوم تصویر
        zoom_frame = cv2.resize(frame[y:y+h, x:x+w], None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)

        # کوچک‌تر کردن تصویر زوم‌شده برای چهره‌های دور
        if zoom_factor <= 1.2:
            zoom_frame = cv2.resize(zoom_frame, (300, 300), interpolation=cv2.INTER_AREA)

        # نمایش تصویر زوم‌شده
        cv2.imshow("زوم هوشمند", zoom_frame)

    # نمایش تصویر اصلی
    cv2.imshow("تشخیص چهره", frame)

    # خروج با فشردن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
