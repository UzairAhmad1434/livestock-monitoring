import cv2
import numpy as np
import time
from ultralytics import YOLO

# Model load karein (Agar system support kare toh yolo12s.pt (Small) use karne se accuracy aur barh jayegi)
model = YOLO("yolo12n.pt") 
cap = cv2.VideoCapture(r"U:\goat counter\sheep.mp4")

w, h, fps_video = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("sheep_counting_accurate.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))

y_line = int(h - 200)

sheep_count = 0
counted_ids = set() 
previous_y = {} 

window_name = "High-Accuracy Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1024, 768)

# --- UI Colors ---
COLOR_LINE = (0, 165, 255)      
COLOR_CROSS = (0, 255, 0)       
COLOR_BBOX = (255, 144, 30)     
COLOR_TEXT = (255, 255, 255)    

line_color = COLOR_LINE
cross_frame_timer = 0  

# --- PiP Setup ---
pip_w, pip_h = 250, 200
last_crop = np.zeros((pip_h, pip_w, 3), dtype=np.uint8)
cv2.putText(last_crop, "Waiting...", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing mukammal ho gayi hai.")
        break

    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    clean_frame = frame.copy() 

    # --- HIGH ACCURACY TRACKING SETTINGS ---
    # tracker="botsort.yaml": IDs ko lose hone se bachata hai
    # imgsz=640: Detection quality behtar karta hai
    # iou=0.5: Overlapping (ek dusre ke upar) sheep ko alag karta hai
    results = model.track(
        frame, 
        persist=True, 
        tracker="botsort.yaml", 
        classes=[18], 
        imgsz=640, 
        conf=0.35, 
        iou=0.5, 
        verbose=False
    )

    # UI Layout
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)        
    cv2.rectangle(overlay, (0, 110), (280, h), (15, 15, 15), -1)   
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    if cross_frame_timer > 0:
        line_color = COLOR_CROSS
        cross_frame_timer -= 1
    else:
        line_color = COLOR_LINE

    cv2.line(frame, (280, y_line), (w, y_line), line_color, 3)
    cv2.line(frame, (280, y_line), (w, y_line), (255, 255, 255), 1)

    current_sheep_in_frame = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        current_sheep_in_frame = len(track_ids)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Bounding Box
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
            
            # ID Label
            label = f"ID: {track_id}"
            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1 - 1, y1 - txt_h - 10), (x1 + txt_w + 5, y1), (0, 0, 0), -1) 
            cv2.rectangle(frame, (x1, y1 - txt_h - 10), (x1 + txt_w + 5, y1), COLOR_BBOX, 1) 
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

            # Robust Line Crossing Logic
            if track_id in previous_y:
                prev_cy = previous_y[track_id] 
                
                # Double check ke sheep ne waqai line mukammal cross ki hai ya nahi
                if (prev_cy < y_line and cy >= y_line) or (prev_cy > y_line and cy <= y_line):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        sheep_count += 1
                        cross_frame_timer = 15 
                        
                        crop_y1, crop_y2 = max(0, y1 - 20), min(h, y2 + 20)
                        crop_x1, crop_x2 = max(0, x1 - 20), min(w, x2 + 20)
                        sheep_crop = clean_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if sheep_crop.size > 0:
                            last_crop = cv2.resize(sheep_crop, (pip_w, pip_h))

            previous_y[track_id] = cy

    # PiP View
    pip_x = w - pip_w - 20 
    pip_y = 130             
    frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = last_crop
    cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_CROSS, 3)
    cv2.rectangle(frame, (pip_x, pip_y - 25), (pip_x + pip_w, pip_y), COLOR_CROSS, -1)
    cv2.putText(frame, "Last Counted", (pip_x + 5, pip_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Sidebar Stats
    cv2.putText(frame, "SYSTEM STATS", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.line(frame, (20, 160), (260, 160), COLOR_BBOX, 2) 
    cv2.putText(frame, f"Live FPS: {fps}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Active Sheep: {current_sheep_in_frame}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(frame, f"Resolution: {w}x{h}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Top Banner Text
    text = f"Total Sheep Count: {sheep_count}"
    cv2.putText(frame, text, (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, COLOR_TEXT, 2, cv2.LINE_AA)

    cv2.imshow(window_name, frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()