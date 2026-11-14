#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// 识别函数（同上）
string detect_color(Mat& frame, Mat& result_img) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat mask_red1, mask_red2, mask_red, mask_yellow, mask_green;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), mask_red1);
    inRange(hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), mask_red2);
    bitwise_or(mask_red1, mask_red2, mask_red);
    inRange(hsv, Scalar(20, 100, 100), Scalar(35, 255, 255), mask_yellow);
    inRange(hsv, Scalar(40, 100, 100), Scalar(85, 255, 255), mask_green);

    double area_red = countNonZero(mask_red);
    double area_yellow = countNonZero(mask_yellow);
    double area_green = countNonZero(mask_green);

    double min_area = 100;
    double max_area = max({area_red, area_yellow, area_green});
    
    string color_name = "UNKNOWN";
    Mat mask;
    Scalar box_color;

    if (max_area < min_area) { frame.copyTo(result_img); return "UNKNOWN"; }
    else if (max_area == area_red) { color_name = "RED"; mask = mask_red; box_color = Scalar(0,0,255); }
    else if (max_area == area_yellow) { color_name = "YELLOW"; mask = mask_yellow; box_color = Scalar(0,255,255); }
    else { color_name = "GREEN"; mask = mask_green; box_color = Scalar(0,255,0); }

    frame.copyTo(result_img);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > min_area) {
            Rect rect = boundingRect(contours[i]);
            rectangle(result_img, rect, box_color, 2);
            putText(result_img, color_name, Point(rect.x, rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
        }
    }
    return color_name;
}

int main() {
    // 1. 打开视频文件 (如果你用摄像头，改成 VideoCapture cap(0); )
    VideoCapture cap("test_video.mp4"); 
    if (!cap.isOpened()) {
        cerr << "无法打开视频！请检查文件名或路径。" << endl;
        return -1;
    }

    Mat frame;
    int frame_count = 0;

    cout << "开始处理视频..." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "视频播放完毕。" << endl;
            break;
        }

        auto start = high_resolution_clock::now();
        Mat result_img;
        string result = detect_color(frame, result_img);
        auto end = high_resolution_clock::now();
        double latency = duration_cast<microseconds>(end - start).count() / 1000.0;

        // 画面显示结果
        string text = result + " (" + to_string((int)latency) + "ms)";
        putText(result_img, text, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

        cout << "帧数: " << frame_count << " | 结果: " << result << " | 耗时: " << latency << " ms" << endl;

        // 每隔30帧保存一张图，方便你查看
        if (frame_count % 30 == 0) {
            string filename = "video_result_" + to_string(frame_count) + ".jpg";
            imwrite(filename, result_img);
        }

        frame_count++;
    }

    cap.release();
    return 0;
}
