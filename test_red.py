import cv2
import numpy as np

# 1. 读取图片
img = cv2.imread('red_light.jpg')
if img is None:
    print("找不到图片，请检查图片名字和路径对不对！")
    exit()

# 2. 转换到 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. 设定红色的 HSV 范围（红色在 HSV 里分两个区间）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 4. 提取红色区域
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 5. 找轮廓画框
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    # 过滤掉太小的噪点
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # 画绿框
        cv2.putText(img, "RED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 6. 不弹窗，直接保存结果图片
cv2.imwrite('result_red.jpg', img)
print("测试完成！请查看当前目录下的 result_red.jpg")
