#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

// 核心识别函数
string detect_color(Mat& img, Mat& result_img) {
    if (img.empty()) return "UNKNOWN";

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 1. 定义颜色范围
    // 红色分两个区间
    Mat mask_red1, mask_red2, mask_red;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), mask_red1);
    inRange(hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), mask_red2);
    bitwise_or(mask_red1, mask_red2, mask_red);

    Mat mask_yellow, mask_green;
    inRange(hsv, Scalar(20, 100, 100), Scalar(35, 255, 255), mask_yellow);
    inRange(hsv, Scalar(40, 100, 100), Scalar(85, 255, 255), mask_green);

    // 2. 计算面积
    double area_red = countNonZero(mask_red);
    double area_yellow = countNonZero(mask_yellow);
    double area_green = countNonZero(mask_green);

    double min_area = 100; // 最小面积阈值
    double max_area = max({area_red, area_yellow, area_green});
    
    string color_name = "UNKNOWN";
    Mat mask;
    Scalar box_color;

    // 3. 判断颜色
    if (max_area < min_area) {
        return "UNKNOWN"; 
    } else if (max_area == area_red) {
        color_name = "RED"; mask = mask_red; box_color = Scalar(0, 0, 255);
    } else if (max_area == area_yellow) {
        color_name = "YELLOW"; mask = mask_yellow; box_color = Scalar(0, 255, 255);
    } else {
        color_name = "GREEN"; mask = mask_green; box_color = Scalar(0, 255, 0);
    }

    // 4. 画框和文字
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    img.copyTo(result_img); // 复制原图用于画框
    
    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > min_area) {
            Rect rect = boundingRect(contours[i]);
            rectangle(result_img, rect, box_color, 2);
            putText(result_img, color_name, Point(rect.x, rect.y - 10), 
                    FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
        }
    }

    return color_name;
}

int main() {
    // 测试的四张图片名字
    vector<string> test_images = {"test_red.jpg", "test_yellow.jpg", "test_green.jpg", "test_unknown.jpg"};

    for (const string& img_path : test_images) {
        cout << "\n正在测试: " << img_path << endl;
        
        Mat img = imread(img_path);
        if (img.empty()) {
            cerr << "[WARN] 无法读取图片: " << img_path << " (请检查文件名或路径)" << endl;
            continue; // 找不到图就跳过，接着测下一张
        }

        Mat result_img;
        
        // 记录开始时间
        auto start_time = high_resolution_clock::now();
        
        // 调用识别函数
        string result = detect_color(img, result_img);
        
        // 记录结束时间并计算耗时
        auto end_time = high_resolution_clock::now();
        double latency = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

        cout << "识别结果: " << result << endl;
        cout << "处理耗时: " << latency << " ms" << endl;

        // 保存结果图
        if (!result_img.empty()) {
            string output_name = "cpp_result_" + img_path;
            imwrite(output_name, result_img);
            cout << "已保存结果图: " << output_name << endl;
        }
    }

    return 0;
}
