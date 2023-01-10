import cv2
import easyocr

prototxt = "modelos/MobileNetSSD_deploy.prototxt.txt"

model = "modelos/MobileNetSSD_deploy.caffemodel"

reader = easyocr.Reader(["es"], gpu=False)

classes = {0:"background", 1:"aeroplane", 2:"bicycle",
          3:"bird", 4:"boat",
          5:"bottle", 6:"bus",
          7:"car", 8:"cat",
          9:"chair", 10:"cow",
          11:"diningtable", 12:"dog",
          13:"horse", 14:"motorbike",
          15:"person", 16:"pottedplant",
          17:"sheep", 18:"sofa",
          19:"train", 20:"tvmonitor"}


net = cv2.dnn.readNetFromCaffe(prototxt, model)

image = cv2.imread("img/IMG_2474.jpg")

height, width, _ = image.shape
image_resized = cv2.resize(image, (300, 300))
# Create a blob
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
print("blob.shape:", blob.shape)

net.setInput(blob)
detections = net.forward()

for detection in detections[0][0]:
     print(detection)
     if detection[2] > 0.45:
          label = classes[detection[1]]

          print("Label:", label)
          box = detection[3:7] * [width, height, width, height]
          x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
          cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
          cv2.putText(image, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
          cv2.putText(image, label, (x_start, y_start - 25), 1, 1.2, (255, 0, 0), 2)
          if label == "car":
            print("ES UN AUTO")
            result = reader.readtext(image)
            for res in result:
                print("result: ", result)
                pt0 = res[0][0]
                pt1 = res[0][1]
                pt2 = res[0][2]
                pt3 = res[0][3]

                print("PPU: "+res[1])
                cv2.rectangle(image, pt0, (pt1[0], pt1[1] - 23), (166,56,242), -1)
                cv2.putText(image, res[1], (pt0[0], pt0[1]- 3), 2 ,0.8,(255,255,255), 1)

                cv2.rectangle(image, pt0, pt2, (0,255,0), 2)

                cv2.circle(image, pt0, 2, (255,0,0),2)
                cv2.circle(image, pt1, 2, (255,0,0),2)
                cv2.circle(image, pt2, 2, (255,0,0),2)
                cv2.circle(image, pt3, 2, (255,0,0),2)
    
  
            
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()