import cv2
import easyocr

reader = easyocr.Reader(["es"], gpu=False)

image = cv2.imread("img/IMG_2474.jpg")
result = reader.readtext(image)

for res in result:
    print("result: ", result)
    pt0 = res[0][0]
    pt1 = res[0][1]
    pt2 = res[0][2]
    pt3 = res[0][3]
    print(res[1])

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