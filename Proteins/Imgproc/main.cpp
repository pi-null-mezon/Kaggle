#include <iostream>

#include <opencv2/opencv.hpp>

#include <dlib/dir_nav.h>
#include <dlib/opencv.h>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"
#include "dlibimgaugment.h"

using namespace std;

cv::Mat __loadImage(const std::string &_filenameprefix,int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded)
{
    cv::Mat _channelsmat[3];
    std::string _postfix[3] = {"_blue.png", "_green.png", "_yellow.png"};
    for(uint8_t i = 0; i < 3; ++i) {
        _channelsmat[i] = loadIFgraymatWsize(_filenameprefix+_postfix[i],_tcols,_trows,_crop,_center,_normalize,_isloadded);
    }
    cv::Mat _outmat;
    cv::merge(_channelsmat,3,_outmat);
    return _outmat;
}

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
        if(file.full_name().find("_green") != string::npos) {
            _filename = file.full_name().substr(0,file.full_name().find("_green"));
            //cv::Mat _mat = loadIFgraymatWsize(file.full_name().substr(),400,400,false,true,true);
            cv::Mat _mat = __loadImage(_filename,512,512,false,true,true,&isloaded);
            cout << "---------------------------" << endl;
            cout << "Filename: " << file.full_name() << endl;
            cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
            cout << "Img channels: " << _mat.channels() << endl;
            cv::Mat _tmpmat;
            if(_mat.empty() == false) {
                for(int j = 0; j < 10; ++j) {

                    _tmpmat = cropimage(_mat,cv::Size(400,400));
                    //_tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(256,256),0.015,0.1,180,cv::BORDER_REFLECT101);
                    if(rnd.get_random_float() > 0.5f)
                        cv::flip(_tmpmat,_tmpmat,0);
                    if(rnd.get_random_float() > 0.5f)
                        cv::flip(_tmpmat,_tmpmat,1);
                    //_tmpmat = jitterimage(_mat,cvrng,cv::Size(0,0),0.015,0.1,180,cv::BORDER_REFLECT101);
                    //_tmpmat = distortimage(_tmpmat,cvrng,0.05,cv::INTER_LANCZOS4, cv::BORDER_REFLECT101);
                    //_tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.4f,0.4f,45.0f*rnd.get_random_float());
                    cv::imshow("Probe", _tmpmat);
                    cv::imshow("Original", _mat);
                    cv::waitKey(0);
                }
            }
        }

    }
    return 0;
}


