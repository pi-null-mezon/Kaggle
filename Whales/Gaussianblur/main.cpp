#include <iostream>

#include <QDir>
#include <QStringList>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

int main()
{
    QDir qdir("/home/alex/Testdata/Flukes/Test");

    QStringList filters;
    filters << "*.jpg" << "*.png" << "*.jpeg";

    QStringList files = qdir.entryList(filters,QDir::Files | QDir::NoDotAndDotDot);
    for(int i = 0; i < files.size(); ++i) {
        cv::String _filename = qdir.absoluteFilePath(files.at(i)).toUtf8().constData();
        cv::Mat _mat = cv::imread(_filename,cv::IMREAD_ANYCOLOR);
        qInfo("%d) %s",i,_filename.c_str());
        if(!_mat.empty()) {
            cv::GaussianBlur(_mat,_mat,cv::Size(9,9),0);
            cv::imwrite(_filename,_mat);
        }

    }
}
