import cv2

cam = cv2.VideoCapture(0)

cam.set(3, 740)
cam.set(4, 580)

Names = []
Namesfile = "coco.names"

with open(Namesfile, "rt" ) as f:
    Names = f.read().rstrip("\n").split("\n")


config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight, config)
net.setInputSize(320 , 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    success, img = cam.read()
    
    classIds, confs , bbox = net.detect(img , confThreshold=0.5)
    print(classIds)
    if len(classIds) !=0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
            cv2.putText(img, Names[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255,0, 0), thickness=2)

    cv2.imshow("camera", img)
    cv2.waitKey(1)