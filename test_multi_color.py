import cv2
import numpy as np
import time

def detect_color(img):
    # 如果图片读取失败
    if img is None:
        return "UNKNOWN", None

    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义各颜色的 HSV 范围
    # 红色分两个区间
    red_ranges = [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ]
    yellow_range = (np.array([20, 100, 100]), np.array([35, 255, 255]))
    green_range = (np.array([40, 100, 100]), np.array([85, 255, 255]))

    # 提取掩膜
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_ranges[0][0], red_ranges[0][1]),
        cv2.inRange(hsv, red_ranges[1][0], red_ranges[1][1])
    )
    mask_yellow = cv2.inRange(hsv, yellow_range[0], yellow_range[1])
    mask_green = cv2.inRange(hsv, green_range[0], green_range[1])

    # 计算各颜色区域面积
    area_red = cv2.countNonZero(mask_red)
    area_yellow = cv2.countNonZero(mask_yellow)
    area_green = cv2.countNonZero(mask_green)

    # 设定一个最小面积阈值（防止噪点误判）
    min_area = 100

    # 找出面积最大的颜色（且必须大于阈值）
    max_area = max(area_red, area_yellow, area_green)
    
    if max_area < min_area:
        return "UNKNOWN", None  # 什么都没找到

    if max_area == area_red:
        color_name = "RED"
        mask = mask_red
        box_color = (0, 0, 255) # 红色画红框
    elif max_area == area_yellow:
        color_name = "YELLOW"
        mask = mask_yellow
        box_color = (0, 255, 255) # 黄色画黄框
    else:
        color_name = "GREEN"
        mask = mask_green
        box_color = (0, 255, 0) # 绿色画绿框

    # 在图上画框和文字
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(img, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    return color_name, img

# ====== 主程序测试 ======
if __name__ == "__main__":
    # 测试的四张图片名字，请确保你的文件夹里有这四张图
    test_images = ['test_red.jpg', 'test_yellow.jpg', 'test_green.jpg', 'test_unknown.jpg']
    
    for img_path in test_images:
        print(f"\n正在测试: {img_path}")
        img = cv2.imread(img_path)
        
        start_time = time.time() # 记录开始时间
        result, result_img = detect_color(img)
        end_time = time.time()   # 记录结束时间
        
        latency = (end_time - start_time) * 1000 # 计算耗时（毫秒）
        
        print(f"识别结果: {result}")
        print(f"处理耗时: {latency:.2f} ms")
        
        # 保存结果图片
        if result_img is not None:
            output_name = f"result_{img_path}"
            cv2.imwrite(output_name, result_img)
            print(f"已保存结果图: {output_name}")
