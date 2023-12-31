import cvzone
from ultralytics import YOLO
import cv2
import math
from sort import *
import random
from collections import defaultdict
cap = cv2.VideoCapture("Videos/walking.mp4")
model = YOLO("../yolov8_weights/yolov8n.pt")
print("YOLOV8. 🚀")
time.sleep(1)
print("YOLOV8.. 🚀")
time.sleep(2)

cap.set(3, 640) # set video width
cap.set(4, 480) # set video height



classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=10)
tracker = Sort(max_age = 20, min_hits = 2, iou_threshold = 0.2)
prev_frame_time = 0
new_frame_time = 0
totalCount = []
error_count = 0
totalcount=[]
known_face_metadata = []

object_id_list = []
centroid = defaultdict(list)
#df = ps.DataFrame(columns=['Class','ID','conf','Time'])
#detections code
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(40)]

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    #img = imutils.resize(img, width=800)
    results = model(img,stream=True)
    detections = np.empty((0, 5))
    h , w, _ , = img.shape
    print(img.shape)
    for b in results:
        boxes = b.boxes
        for box in boxes:
            total = len(boxes)
            #print(total)
            x1, y1, x2, y2 = box.xyxy[0]
            #print(box.xyxy[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 -x1 , y2 - y1

            conf = math.ceil((box.conf[0] * 100))/100
            #print(conf)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if totalcount.count(total) == 0:
                totalcount.append(total)
            if len(str(totalcount)) >= 5:
                    # cv2.rectangle(img,(0,0),(1280,720),(0,0,255),10)
                    # print("FULL!)
                #cvzone.putTextRect(img, f' FULL!:{len(totalcount)}', (0, 300), colorT=(0, 0, 255),
                                       #colorR=(131, 139, 139))
             if currentClass == "person" and conf < 0.9:
                cv2.putText(img, f'  -{conf}', (x1, y1 - 10),
                             cv2.FONT_HERSHEY_DUPLEX,
                             1, (0, 0, 255), 1)
                cvzone.putTextRect(img, f' Number of people:{total}', (0, 50), colorT=(0, 255, 0), colorR=(131, 139, 139))

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    Tracker = tracker.update(detections)
#tracker code
    for result in Tracker:
        x1, y1, x2, y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #print(result)
        cv2.rectangle(img, (x1, y1), (x2, y2), (colors[int(id) % len(colors)]), 2)
        cv2.putText(img,f'{int(id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0,255 ), 1)
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        cv2.circle(img, (cx, cy), 4,(colors[int(id) % len(colors)]), -1)

        centroid[id].append((cx, cy))
        if id not in object_id_list:
            object_id_list.append(id)
            start_pt = (cx, cy)
            end_pt = (cx, cy)
            cv2.line(img, start_pt, end_pt, (colors[int(id) % len(colors)]), 2)
        else:
            l = len(centroid[id])
            for p in range(len(centroid[id])):
                if not p + 1 == l:
                    start_pt = (centroid[id][p][0], centroid[id][p][1])
                    end_pt = (centroid[id][p + 1][0], centroid[id][p + 1][1])
                    cv2.line(img, start_pt, end_pt, (colors[int(id) % len(colors)]), 1)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("FPS:",fps)

    cv2.imshow("Webcam",img)
    if cv2.waitKey(10) & 0xff == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
