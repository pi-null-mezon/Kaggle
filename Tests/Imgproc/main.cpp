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
        cv::Mat _mat = loadIbgrmatWsize(_filename,512,512,false,&isloaded);
        assert(isloaded);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        cv::Mat _tmpmat;
        if(_mat.empty() == false) {
            for(int j = 0; j < 10; ++j) {

                _tmpmat = _mat.clone();

                /*if(rnd.get_random_float() > 0.5f)
                    cv::rotate(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.5f)
                    cv::rotate(_tmpmat,_tmpmat,0);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);
                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,0);*/

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.03,0.04,0,cv::BORDER_REFLECT,cv::Scalar(0),false);
                /*if(rnd.get_random_float() > 0.1f)
                    _tmpmat = distortimage(_tmpmat,cvrng,0.03,cv::INTER_CUBIC,cv::BORDER_WRAP,cv::Scalar(0));

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),0,0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,rnd.get_random_float(),1,0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,0,rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);
                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = cutoutRect(_tmpmat,1,rnd.get_random_float(),0.3f,0.3f,rnd.get_random_float()*180.0f);

                if(rnd.get_random_float() > 0.5f)
                    cv::blur(_tmpmat,_tmpmat,cv::Size(3,3));

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat *= (0.8 + 0.4*rnd.get_random_double());

                if(rnd.get_random_float() > 0.1f)
                    _tmpmat = addNoise(_tmpmat,cvrng,0,7);*/

                dlib::matrix<dlib::rgb_pixel> _dlibtmpimg = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
                dlib::disturb_colors(_dlibtmpimg,rnd);

                cv::imshow("Probe", dlibmatrix2cvmat(_dlibtmpimg));
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }
    }
    return 0;
}


