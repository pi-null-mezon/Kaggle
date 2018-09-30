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
    dlib::rand rnd(time(nullptr));
    cv::RNG    cvrng(static_cast<uint64>(time(nullptr)));

    for(auto file : dir.get_files()) {

        cv::Mat _mat = loadIbgrmatWsize(file.full_name(),360,270,true);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        if(_mat.empty() == false) {           
            for(int j = 0; j < 10; ++j) {
                cv::Mat _tmpmat = jitterimage(_mat,cvrng,cv::Size(0,0),0.17,0.02,15,cv::BORDER_REPLICATE);
                if(rnd.get_random_float() > 0.5f) {
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0);
                } else {
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1);
                }
                if(rnd.get_random_float() > 0.5f) {
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float());
                } else {
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float());
                }
                cv::imshow("Probe", _tmpmat);
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }

    }
    return 0;
}
