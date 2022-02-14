#include <iostream>

#include <opencv2/opencv.hpp>

#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "opencvlbpoperator.h"
#include "dlibimgaugment.h"
#include "opencvhistograms.h"

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

    std::string _filename;
    bool isloaded;
    for(auto file : dir.get_files()) {
        _filename = file.full_name();
        cv::Mat _mat = loadIbgrmatWsize(_filename,150,150,false,&isloaded);
        assert(isloaded);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << " - dimensions: " << _mat.cols << "x" << _mat.rows << endl;
        cout << " - depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << " - channels: " << _mat.channels() << endl;
        cv::Mat _tmpmat;
        if(_mat.empty() == false) {
            for(int j = 0; j < 1; ++j) {

                _tmpmat = colors_histograms(_mat);

                cv::namedWindow("Probe", cv::WINDOW_GUI_EXPANDED);
                cv::imshow("Probe", cv::lbph::localNormalization(_tmpmat));
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }
    }
    return 0;
}


