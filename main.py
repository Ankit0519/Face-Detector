print("MY NEW CODE IS RUNNING")
import cv2
import pygame
from ultralytics import YOLO
import mediapipe as mp

# =====================================
# 1. AUDIO FILES
# =====================================
PHONE_ALARM = "phone_alarm.mp3"
EYE_ALARM = "eye_alarm.mp3"
NO_FACE_ALARM = "no_face_alarm.mp3"

# =====================================
# 2. PHONE DETECTION MODEL
# =====================================
print("Phone detection model load ho raha hai...")
phone_model = YOLO("yolo11n.pt")

# =====================================
# 3. FACE AND EYE LANDMARK SETUP
# =====================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# 4. AUDIO SETUP
# =====================================
pygame.mixer.init()
current_alarm = None

# =====================================
# 5. CAMERA SETUP
# =====================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera open nahi hua!")
    face_mesh.close()
    pygame.mixer.quit()
    raise SystemExit

print("EyeFaceAlarm AI start ho gaya!")
print("Band karne ke liye Q dabao.")

# =====================================
# 6. MAIN DETECTION LOOP
# =====================================
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera se video nahi mil raha.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # ----- FACE AND EYE LANDMARKS -----
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        face_detected = (
            face_results.multi_face_landmarks is not None
        )
        eyes_detected = False

        if face_detected:
            landmarks = (
                face_results.multi_face_landmarks[0].landmark
            )

            # Approximate eye landmark points
            eye_points = [33, 133, 159, 145, 362, 263, 386, 374]

            for point in eye_points:
                x = int(landmarks[point].x * width)
                y = int(landmarks[point].y * height)

                cv2.circle(
                    frame, (x, y), 2, (0, 255, 0), -1
                )

            # Landmark visibility; actual eye-closure detection
            # is not implemented in this version.
            eyes_detected = True

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "EYE LANDMARKS FOUND",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----- PHONE DETECTION -----
        phone_detected = False

        phone_results = phone_model.predict(
            frame,
            imgsz=640,
            conf=0.45,
            verbose=False
        )

        for result in phone_results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                # COCO class 67 = cell phone
                if class_id == 67:
                    phone_detected = True

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "PHONE DETECTED",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

        # ----- ALARM SELECTION -----
        # Priority: phone > missing face > eye landmarks
        if phone_detected:
            desired_alarm = PHONE_ALARM
        elif not face_detected:
            desired_alarm = NO_FACE_ALARM
        elif not eyes_detected:
            desired_alarm = EYE_ALARM
        else:
            desired_alarm = None

        # Play/restart only when alarm type changes
        if desired_alarm != current_alarm:
            pygame.mixer.music.stop()
            current_alarm = desired_alarm

            if current_alarm is not None:
                try:
                    pygame.mixer.music.load(current_alarm)
                    pygame.mixer.music.play(-1)
                except pygame.error as error:
                    print("Audio error:", error)

        # ----- DISPLAY CAMERA -----
        cv2.imshow("EyeFaceAlarm AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    cv2.destroyAllWindows()
    print("EyeFaceAlarm AI band ho gaya.")
