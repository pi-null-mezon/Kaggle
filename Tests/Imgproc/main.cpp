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

    std::string _filename;
    bool isloaded;
    for(auto file : dir.get_files()) {
        _filename = file.full_name();
        cv::Mat _mat = loadIFgraymatWsize(_filename,512,192,false,true,false,&isloaded);
        assert(isloaded);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        cv::Mat _tmpmat;
        dlib::matrix<float> _dlibmatrix;
        if(_mat.empty() == false) {
            for(int j = 0; j < 10; ++j) {

                _tmpmat = _mat.clone();/*jitterimage(_mat,cvrng,cv::Size(0,0),0.05,0.05,5,cv::BORDER_REFLECT101,true);*/
                /*if(rnd.get_random_float() > 0.5f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.02,cv::INTER_CUBIC,cv::BORDER_REFLECT101);*/

                /*if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0.25f + 0.5f*rnd.get_random_float(),0,0.2f,0.4f,rnd.get_random_float()*180.0f);*/
                /*if(rnd.get_random_float() > 0.05f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0,0.75f + 0.25f*rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,1,0.75f + 0.25f*rnd.get_random_float(),0.2f,0.4f,rnd.get_random_float()*180.0f);*/

                /*if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutEllipse(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.1f,0.3f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.5f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));*/
                cv::flip(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.5f)
                    _tmpmat = addNoise(_tmpmat,cvrng,0.1f*rnd.get_random_float()-0.025f,0.1f*rnd.get_random_float());

                if(rnd.get_random_float() > 0.5f)
                    _tmpmat = -1.0f*_tmpmat;

                /*if(rnd.get_random_float() > 0.5f)
                    _tmpmat *= 1.0f + 1.0f*rnd.get_random_float();*/

                //_tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),rnd.get_random_float(),0.2f,0.4f,180.0f*rnd.get_random_float());
                /*_dlibmatrix = cvmat2dlibmatrix<float>(_tmpmat);
                dlib::disturb_colors(_dlibmatrix,rnd);
                _tmpmat = dlibmatrix2cvmat<float>(_dlibmatrix);*/
                cv::imshow("Probe", _tmpmat);
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }
    }
    return 0;
}


