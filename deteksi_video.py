import cv2 as cv
import time
import geocoder
import os

# baca label
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# load model YOLOV4-TINY (CPU) 
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# input video
cap = cv.VideoCapture("asset/video/vid.mp4")
if not cap.isOpened():
    print("Gagal membuka video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Gagal membaca frame pertama.")
    exit()

height, width = frame.shape[:2]

# output video
output_video_path = "output/video/output_hasil.mp4"

result = cv.VideoWriter(output_video_path,
                        cv.VideoWriter_fourcc(*'mp4v'),
                        10, (width, height))

save_path = "output/video"
g = geocoder.ip('me')
i = 0                 
last_saved = 0 

# deteksi parameter
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
starting_time = time.time()   

# loop deteksi
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)

    detected = False

    for (classid, score, box) in zip(classes, scores, boxes):
        label = class_name[classid] if classid < len(class_name) else "unknown"
        x, y, w, h = box
        recarea = w * h
        area = width * height

        if score >= 0.7 and (recarea / area) <= 0.1 and y < 600:
            detected = True
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{label}: {score:.2f}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # jika terdeteksi & sudah lewat 2 detik, simpan
    current_time = time.time()
    if detected and (current_time - last_saved) >= 2:
        image_path = os.path.join(save_path, f"pothole{i}.jpg")
        text_path = os.path.join(save_path, f"pothole{i}.txt")

        cv.imwrite(image_path, frame)
        with open(text_path, 'w') as f:
            f.write(str(g.latlng))
        i += 1
        last_saved = current_time

    # hitung FPS dan tampilkan
    elapsed = current_time - starting_time
    fps = frame_counter / elapsed
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # tampilkan & simpan video
    cv.imshow('frame', frame)
    result.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# selesai
cap.release()
result.release()
cv.destroyAllWindows()
