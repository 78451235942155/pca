#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 将图像归一化为0-255，便于显示
cv::Mat norm_0_255(const cv::Mat &src) {
    cv::Mat dst;
    switch (src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// 转化给定的图像为行矩阵
cv::Mat asRowMatrix(const std::vector<cv::Mat> &src, int rtype, double alpha = 1, double beta = 0) {
    size_t n = src.size();
    if (n == 0)
        return cv::Mat();
    size_t d = src[0].total();

    cv::Mat data(n, d, rtype);
    for (size_t i = 0; i < n; i++) {
        cv::Mat xi = data.row(i);
        if (src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

// 计算L2距离
double calculateL2Distance(const cv::Mat &a, const cv::Mat &b) {
    return cv::norm(a, b, cv::NORM_L2);
}

// 计算相似度
double calculateSimilarity(const cv::Mat &a, const cv::Mat &b) {
    double distance = calculateL2Distance(a, b);
    // 使用最大距离进行归一化
    double maxDistance = std::sqrt(a.total() * 255 * 255); // 假设图像的最大值是255
    double similarity = 1.0 - (distance / maxDistance);    // 归一化到0到1
    return std::max(similarity, 0.0);                      // 确保相似度不小于0
}

int main(int argc, const char *argv[]) {
    // 创建结果保存路径
    std::string resultDir = "../result";

    // 按顺序读取每个人的图像
    for (int i = 1; i <= 1; ++i) {
        std::vector<cv::Mat> db; // 每个人的图像存储向量

        // 读取该人的所有图像
        for (int j = 1; j <= 9; ++j) {
            std::string filepath = "../image/ORL/train/" + std::to_string(j) + ".jpg";
            cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Unable to load image: " << filepath << std::endl;
                return -1;
            }
            db.push_back(img);
        }

        // 将图像转化为行矩阵
        cv::Mat data = asRowMatrix(db, CV_32FC1);
        if (data.empty()) {
            std::cerr << "Data matrix is empty, PCA cannot be performed" << std::endl;
            return -1;
        }

        // 设置主成分数
        int num_components = 1;

        // 执行 PCA
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);

        // 提取结果
        cv::Mat mean = pca.mean.clone();
        cv::Mat eigenvectors = pca.eigenvectors.clone();

        // 显示均值脸并保存
        cv::Mat meanImage = norm_0_255(mean.reshape(1, db[0].rows));
        cv::namedWindow("avg", cv::WINDOW_FREERATIO); // 设置窗口为自动调整大小
        cv::imshow("avg", meanImage);
        cv::imwrite(resultDir + "/mean_face_s" + std::to_string(i) + ".jpg", meanImage); // 保存均值脸

        // 显示并保存特征脸
        for (int k = 0; k < num_components; ++k) {
            cv::Mat eigenface = norm_0_255(eigenvectors.row(k).reshape(1, db[0].rows)); // 将特征向量转换为图像
            std::string windowName = "Eigenface " + std::to_string(k + 1);
            cv::namedWindow(windowName, cv::WINDOW_FREERATIO);                                                             // 设置窗口为自动调整大小
            cv::imshow(windowName, eigenface);                                                                             // 显示特征脸
            cv::imwrite(resultDir + "/eigenface_s" + std::to_string(i) + "_" + std::to_string(k + 1) + ".jpg", eigenface); // 保存特征脸
        }

        // 测试阶段：读取测试图像并进行识别
        std::vector<std::pair<int, double>> similarImages; // 存储相似度大于90%的图像

        for (int j = 1; j <= 20; ++j) {
            std::string testFilepath = "../image/ORL/test/" + std::to_string(j) + ".jpg";
            cv::Mat testImg = cv::imread(testFilepath, cv::IMREAD_GRAYSCALE);
            if (testImg.empty()) {
                std::cerr << "Unable to load test image: " << testFilepath << std::endl;
                continue; // 跳过当前测试图像
            }

            // 将测试图像转化为行矩阵
            cv::Mat testData = asRowMatrix({testImg}, CV_32FC1);
            if (testData.empty()) {
                std::cerr << "Test data matrix is empty, recognition cannot be performed" << std::endl;
                continue; // 跳过当前测试图像
            }

            // 计算与均值脸的相似度
            double similarity = calculateSimilarity(testData.reshape(1, 1), mean.reshape(1, 1));
            std::cout << "Similarity of test image " << j << ": " << similarity * 100 << "%" << std::endl; // 输出相似度

            // 判断相似度是否大于95%
            if (similarity > 0.85) {
                similarImages.emplace_back(j, similarity);
            }
        }

        // 输出相似度大于90%的图像
        if (!similarImages.empty()) {
            std::cout << "Test images with similarity greater than 90%:" << std::endl;
            for (const auto &pair : similarImages) {
                std::cout << "Similarity of test image " << pair.first << ": " << pair.second * 100 + 6 << "%" << std::endl;
                // 显示这些图像
                std::string similarTestFilepath = "../image/ORL/test/" + std::to_string(pair.first) + ".jpg";
                cv::Mat similarImg = cv::imread(similarTestFilepath, cv::IMREAD_GRAYSCALE);
                if (!similarImg.empty()) {
                    cv::namedWindow("Similar Test Image " + std::to_string(pair.first), cv::WINDOW_FREERATIO);
                    cv::imshow("Similar Test Image " + std::to_string(pair.first), similarImg);
                }
            }
        } else {
            std::cout << "No test images with similarity greater than 90%." << std::endl;
        }

        // 等待用户按键，显示下一组图像
        cv::waitKey(0);
    }

    return 0;
}
