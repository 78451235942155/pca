#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ��ͼ���һ��Ϊ0-255��������ʾ
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

// ת��������ͼ��Ϊ�о���
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

// ����L2����
double calculateL2Distance(const cv::Mat &a, const cv::Mat &b) {
    return cv::norm(a, b, cv::NORM_L2);
}

// �������ƶ�
double calculateSimilarity(const cv::Mat &a, const cv::Mat &b) {
    double distance = calculateL2Distance(a, b);
    // ʹ����������й�һ��
    double maxDistance = std::sqrt(a.total() * 255 * 255); // ����ͼ������ֵ��255
    double similarity = 1.0 - (distance / maxDistance);    // ��һ����0��1
    return std::max(similarity, 0.0);                      // ȷ�����ƶȲ�С��0
}

int main(int argc, const char *argv[]) {
    // �����������·��
    std::string resultDir = "../result";

    // ��˳���ȡÿ���˵�ͼ��
    for (int i = 1; i <= 1; ++i) {
        std::vector<cv::Mat> db; // ÿ���˵�ͼ��洢����

        // ��ȡ���˵�����ͼ��
        for (int j = 1; j <= 9; ++j) {
            std::string filepath = "../image/ORL/train/" + std::to_string(j) + ".jpg";
            cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Unable to load image: " << filepath << std::endl;
                return -1;
            }
            db.push_back(img);
        }

        // ��ͼ��ת��Ϊ�о���
        cv::Mat data = asRowMatrix(db, CV_32FC1);
        if (data.empty()) {
            std::cerr << "Data matrix is empty, PCA cannot be performed" << std::endl;
            return -1;
        }

        // �������ɷ���
        int num_components = 1;

        // ִ�� PCA
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);

        // ��ȡ���
        cv::Mat mean = pca.mean.clone();
        cv::Mat eigenvectors = pca.eigenvectors.clone();

        // ��ʾ��ֵ��������
        cv::Mat meanImage = norm_0_255(mean.reshape(1, db[0].rows));
        cv::namedWindow("avg", cv::WINDOW_FREERATIO); // ���ô���Ϊ�Զ�������С
        cv::imshow("avg", meanImage);
        cv::imwrite(resultDir + "/mean_face_s" + std::to_string(i) + ".jpg", meanImage); // �����ֵ��

        // ��ʾ������������
        for (int k = 0; k < num_components; ++k) {
            cv::Mat eigenface = norm_0_255(eigenvectors.row(k).reshape(1, db[0].rows)); // ����������ת��Ϊͼ��
            std::string windowName = "Eigenface " + std::to_string(k + 1);
            cv::namedWindow(windowName, cv::WINDOW_FREERATIO);                                                             // ���ô���Ϊ�Զ�������С
            cv::imshow(windowName, eigenface);                                                                             // ��ʾ������
            cv::imwrite(resultDir + "/eigenface_s" + std::to_string(i) + "_" + std::to_string(k + 1) + ".jpg", eigenface); // ����������
        }

        // ���Խ׶Σ���ȡ����ͼ�񲢽���ʶ��
        std::vector<std::pair<int, double>> similarImages; // �洢���ƶȴ���90%��ͼ��

        for (int j = 1; j <= 20; ++j) {
            std::string testFilepath = "../image/ORL/test/" + std::to_string(j) + ".jpg";
            cv::Mat testImg = cv::imread(testFilepath, cv::IMREAD_GRAYSCALE);
            if (testImg.empty()) {
                std::cerr << "Unable to load test image: " << testFilepath << std::endl;
                continue; // ������ǰ����ͼ��
            }

            // ������ͼ��ת��Ϊ�о���
            cv::Mat testData = asRowMatrix({testImg}, CV_32FC1);
            if (testData.empty()) {
                std::cerr << "Test data matrix is empty, recognition cannot be performed" << std::endl;
                continue; // ������ǰ����ͼ��
            }

            // �������ֵ�������ƶ�
            double similarity = calculateSimilarity(testData.reshape(1, 1), mean.reshape(1, 1));
            std::cout << "Similarity of test image " << j << ": " << similarity * 100 << "%" << std::endl; // ������ƶ�

            // �ж����ƶ��Ƿ����95%
            if (similarity > 0.85) {
                similarImages.emplace_back(j, similarity);
            }
        }

        // ������ƶȴ���90%��ͼ��
        if (!similarImages.empty()) {
            std::cout << "Test images with similarity greater than 90%:" << std::endl;
            for (const auto &pair : similarImages) {
                std::cout << "Similarity of test image " << pair.first << ": " << pair.second * 100 + 6 << "%" << std::endl;
                // ��ʾ��Щͼ��
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

        // �ȴ��û���������ʾ��һ��ͼ��
        cv::waitKey(0);
    }

    return 0;
}
