import cv2 as cv
import os

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
this_path=os.getcwd()
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
#print(class_name)

#get the yolo model weights & params
net = cv.dnn.readNet('yolov7-tiny.weights', 'yolov7-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture(f'{this_path}/dimaria.mp4')
#cap=cv.VideoCapture(0)
#get the dimensions for saving the video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv.VideoWriter('world_cup.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        #color = COLORS[int(classid) % len(COLORS)]
        color=(0,0,0)
        label = "%s : %f" %(class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    
    result.write(frame)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
