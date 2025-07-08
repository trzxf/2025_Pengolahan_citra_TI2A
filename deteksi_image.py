import cv2
import os

folder_path = "asset/foto"
output_folder = "output/foto"

# Load label dan model
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# Loop semua file gambar
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Gagal membaca: {filename}")
            continue

        classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)

        cv2.imshow("Hasil Deteksi", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        result_path = os.path.join(output_folder, f"detected_{filename}")
        cv2.imwrite(result_path, img)
        print(f"Hasil disimpan: {result_path}")