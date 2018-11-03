
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


#define EPS  1e-5


struct FaceBox
{
    int x, y, w, h;
    float angle, scale, score;

    Rect getBox() { return Rect(x,y,w,h); }

    FaceBox(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
    : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), score(c_)
    {}
};


bool xyValid(int _x, int _y, cv::Mat _img)
{
    if (_x >= 0 && _x < _img.cols && _y >= 0 && _y < _img.rows)
        return true;
    else
        return false;
}


cv::Mat preprocessImg(cv::Mat _img)
{
    cv::Mat mean(_img.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    _img.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat preprocessImg(cv::Mat _img, int _dim)
{
    cv::Mat imgNew;
    cv::resize(_img, imgNew, cv::Size(_dim, _dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}


cv::Mat resizeImg(cv::Mat _img, float _scale)
{
    cv::Mat aux;
    cv::resize(_img, aux, cv::Size(int(_img.cols / _scale), int(_img.rows / _scale)));
    return aux;
}

cv::Mat padImg(cv::Mat _img)
{
    int row = std::min(int(_img.rows * 0.2), 100);
    int col = std::min(int(_img.cols * 0.2), 100);
    cv::Mat aux;
    cv::copyMakeBorder(_img, aux, row, row, col, col, cv::BORDER_CONSTANT);
    return aux;
}


bool compareFaceByScore(const FaceBox &box1, const FaceBox &box2)
{
    return box1.score > box2.score;
}


float IoU(FaceBox &box1, FaceBox &box2)
{
    int xOverlap = std::max(0, std::min(box1.x + box1.w - 1, box2.x + box2.w - 1) - std::max(box1.x, box2.x) + 1);
    int yOverlap = std::max(0, std::min(box1.y + box1.h - 1, box2.y + box2.h - 1) - std::max(box1.y, box2.y) + 1);
    int intersection = xOverlap * yOverlap;
    int unio = box1.w * box1.h + box2.w * box2.h - intersection;

    return float(intersection) / unio;
}


std::vector<FaceBox> NMS(std::vector<FaceBox> &_faces, bool _local, float _threshold)
{
    if (_faces.size() == 0)
        return _faces;

    std::sort(_faces.begin(), _faces.end(), compareFaceByScore);
    bool flag[_faces.size()];

    memset(flag, 0, _faces.size());
    for (int i = 0; i < _faces.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < _faces.size(); j++)
        {
            if (_local && abs(_faces[i].scale - _faces[j].scale) > EPS)
                continue;

            if (IoU(_faces[i], _faces[j]) > _threshold)
                flag[j] = 1;
        }
    }

    std::vector<FaceBox> faces_nms;
    for (int i = 0; i < _faces.size(); i++)
    {
        if (!flag[i]) faces_nms.push_back(_faces[i]);
    }
    return faces_nms;
}

std::vector<FaceBox> PCN_1(cv::Mat _img, cv::Mat _paddedImg, cv::dnn::Net _net, float _thresh)
{
    std::vector<FaceBox> faceBoxes_1;
    int row = (_paddedImg.rows - _img.rows) / 2;
    int col = (_paddedImg.cols - _img.cols) / 2;
    int netSize = 24;
    int minFace = 20 * 1.4; // - size 20 + 40%
    float currentScale = minFace / float(netSize);
    int stride = 8;

    cv::Mat resizedImg = resizeImg(_img, currentScale);

    while (std::min(resizedImg.rows, resizedImg.cols) >= netSize)
    {
        // - Set input for net
        Mat inputMat = preprocessImg(resizedImg);
        Mat inputBlob = blobFromImage(inputMat, 1.0, Size(), Scalar(), false, false);
        _net.setInput(inputBlob);

        std::vector<String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_1" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresData = outputBlobs[0];
        cv::Mat rotateProbsData = outputBlobs[1];
        cv::Mat regressionData = outputBlobs[2];

        // scoresData.ptr<float>(0, 1)  ---->  image 0, channel 1
        Mat scoresMat(scoresData.size[2], scoresData.size[3], CV_32F, scoresData.ptr<float>(0, 1));
        Mat reg_1_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 0));
        Mat reg_2_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 1));
        Mat reg_3_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 2));
        Mat rotateProbsMat(rotateProbsData.size[2], rotateProbsData.size[3], CV_32F, rotateProbsData.ptr<float>(0, 1));

        float w = netSize * currentScale;

        for (int i = 0; i < scoresData.size[2]; i++)
        {
            for (int j = 0; j < scoresData.size[3]; j++)
            {
                if (scoresMat.at<float>(i, j) < _thresh)
                    {continue;}

                float score = scoresMat.at<float>(i, j);
                float sn = reg_1_Mat.at<float>(i, j);
                float xn = reg_2_Mat.at<float>(i, j);
                float yn = reg_3_Mat.at<float>(i, j);

                int x = int(j * currentScale * stride - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col;
                int y = int(i * currentScale * stride - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row;
                int wh = int(w * sn);

                if(xyValid(x, y, _paddedImg) && xyValid(x+wh-1, y+wh-1, _paddedImg))
                {
                    if (rotateProbsMat.at<float>(i, j) > 0.5)
                    {
                        faceBoxes_1.push_back(FaceBox(x,y,wh,wh,0, currentScale, score));
                        //rectangle(_paddedImg, Rect(x, y, wh, wh), Scalar(0, 255, 0), 2);
                    }
                    else
                    {
                        faceBoxes_1.push_back(FaceBox(x,y,wh,wh,180, currentScale, score));
                        //rectangle(_paddedImg, Rect(x, y, wh, wh), Scalar(255, 0, 0), 2);
                    }
                }


            }
        }

        resizedImg = resizeImg(resizedImg, currentScale);
        currentScale = float(_img.rows) / resizedImg.rows;
    }

    cout << faceBoxes_1.size() << endl;
    return faceBoxes_1;
}


std::vector<FaceBox> PCN_2(cv::Mat _img, cv::Mat _img180, cv::dnn::Net _net, float _threshold, int _dim, std::vector<FaceBox> _faces)
{
    //_dim = 24 ---> network size
    if (_faces.size() == 0)
        return _faces;

    std::vector<cv::Mat> dataList;
    int height = _img.rows;
    for (int i = 0; i < _faces.size(); i++)
    {
        if (abs(_faces[i].angle) < EPS)
            dataList.push_back(preprocessImg(_img(_faces[i].getBox()), _dim));
        else
        {
            int y2 = _faces[i].y + _faces[i].h - 1;
            dataList.push_back(preprocessImg(_img180(cv::Rect(_faces[i].x, height - 1 - y2, _faces[i].w, _faces[i].h)), _dim));
        }
    }

    std::vector<FaceBox> faceBoxes_2;

    // - Process the dataList vector
    for(int dataNr=0; dataNr<dataList.size(); dataNr++)
    {


        Mat inputBlob = blobFromImage(dataList[dataNr], 1.0, Size(), Scalar(), false, false);
        cout << "Input Data: " << endl;
        cout << "I: " << inputBlob.size[0] << endl;
        cout << "C: " << inputBlob.size[1] << endl;
        cout << "H: " << inputBlob.size[2] << endl;
        cout << "W: " << inputBlob.size[3] << endl << endl;

        _net.setInput(inputBlob);
        std::vector<String> outputBlobNames = {"cls_prob", "rotate_cls_prob", "bbox_reg_2" };
        std::vector<cv::Mat> outputBlobs;

        _net.forward(outputBlobs, outputBlobNames);
        cv::Mat scoresData = outputBlobs[0];
        cv::Mat rotateProbsData = outputBlobs[1];
        cv::Mat regressionData = outputBlobs[2];

        cout << "Scores Data: " << endl;
        cout << "I: " << scoresData.size[0] << endl;
        cout << "C: " << scoresData.size[1] << endl;
        cout << "H: " << scoresData.size[2] << endl;
        cout << "W: " << scoresData.size[3] << endl << endl;

        cout << "Regression Data: " << endl;
        cout << "I: " << regressionData.size[0] << endl;
        cout << "C: " << regressionData.size[1] << endl;
        cout << "H: " << regressionData.size[2] << endl;
        cout << "W: " << regressionData.size[3] << endl << endl;

        cout << "Rotate Data: " << endl;
        cout << "I: " << rotateProbsData.size[0] << endl;
        cout << "C: " << rotateProbsData.size[1] << endl;
        cout << "H: " << rotateProbsData.size[2] << endl;
        cout << "W: " << rotateProbsData.size[3] << endl << endl << endl;


        // scoresData.ptr<float>(0, 1)  ---->  image 0, channel 1
        Mat scoresMat(scoresData.size[2], scoresData.size[3], CV_32F, scoresData.ptr<float>(0, 1));
        Mat reg_1_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 0));
        Mat reg_2_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 1));
        Mat reg_3_Mat(regressionData.size[2], regressionData.size[3], CV_32F, regressionData.ptr<float>(0, 2));

        if(scoresMat.at<float>(0,0) < _threshold)
            continue;

        float score = scoresMat.at<float>(0,0);
        float sn = reg_1_Mat.at<float>(0,0);// reg->data_at(i, 0, 0, 0);
        float xn = reg_2_Mat.at<float>(0,0);// reg->data_at(i, 1, 0, 0);
        float yn = reg_3_Mat.at<float>(0,0);//reg->data_at(i, 2, 0, 0);
        int cropX = _faces[dataNr].x;
        int cropY = _faces[dataNr].y;
        int cropW = _faces[dataNr].w;
        if (abs(_faces[dataNr].angle)  > EPS)
            cropY = height - 1 - (cropY + cropW - 1);
        int w = int(sn * cropW);
        int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
        int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
        float maxRotateScore = 0;
        int maxRotateIndex = 0;
        for (int j = 0; j < 3; j++)
        {
            Mat rotateProbsMat(rotateProbsData.size[2], rotateProbsData.size[3], CV_32F, rotateProbsData.ptr<float>(0, j));
            if (rotateProbsMat.at<float>(0,0) > maxRotateScore)
            {
                maxRotateScore = rotateProbsMat.at<float>(0,0);//rotateProb->data_at(i, j, 0, 0);
                maxRotateIndex = j;
            }
        }
        if(xyValid(x, y, dataList[dataNr]) && xyValid(x+w-1, y+w-1, dataList[dataNr])) //(Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
        {
            float angle = 0;
            if (abs(_faces[dataNr].angle)  < EPS)
            {
                if (maxRotateIndex == 0)
                    angle = 90;
                else if (maxRotateIndex == 1)
                    angle = 0;
                else
                    angle = -90;
                faceBoxes_2.push_back(FaceBox(x,y,w,w,angle, _faces[dataNr].scale, score));
                //ret.push_back(Window2(x, y, w, w, angle, _faces[i].scale, prob->data_at(i, 1, 0, 0)));
            }
            else
            {
                if (maxRotateIndex == 0)
                    angle = 90;
                else if (maxRotateIndex == 1)
                    angle = 180;
                else
                    angle = -90;
                faceBoxes_2.push_back(FaceBox(x, height - 1 -  (y + w - 1), w, w, angle, _faces[dataNr].scale, score));
            }
        }
    }
    return faceBoxes_2;
}

// ------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    Net net_1 = readNet("pcn_model/PCN-1.prototxt", "pcn_model/PCN.caffemodel");
    Net net_2 = readNet("pcn_model/PCN-2.prototxt", "pcn_model/PCN.caffemodel");


    Mat img = imread("imgs/9.jpg");
    Mat paddedImg = padImg(img);

    cv::Mat img180, img90, imgNeg90;
    cv::flip(paddedImg, img180, 0);
    cv::transpose(paddedImg, img90);
    cv::flip(img90, imgNeg90, 0);

    float thresholds[] = {0.37, 0.43, 0.95};

    std::vector<FaceBox> faces = PCN_1(img, paddedImg, net_1, thresholds[0]);
    faces = NMS(faces, true, 0.8);

    faces = PCN_2(paddedImg, img180, net_2, thresholds[1], 24, faces);


    return 1;

}

