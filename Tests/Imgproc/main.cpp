#include <iostream>

#include <opencv2/opencv.hpp>

#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "dlibimgaugment.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "This application should be ran with path to images directory as cmd arg" << endl;
        return 0;
    }

    auto dir = dlib::directory(argv[1]);
    dlib::rand rnd(time(0));
    cv::RNG    cvrng(time(0));

    for(auto file : dir.get_files()) {

        cv::Mat _mat = loadIbgrmatWsize(file.full_name(),300,300,true);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        if(_mat.empty() == false) {           
            for(int j = 0; j < 10; ++j) {
                cv::Mat _tmpmat = jitterimage(_mat,cvrng,cv::Size(0,0),0.2,0.02,10,cv::BORDER_REPLICATE);
                cv::imshow("Probe", _tmpmat);
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }

    }
    return 0;
}
