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
    cv::RNG cvrng(static_cast<uint64>(time(nullptr)));

    std::string _filename;
    bool isloaded;
    for(auto file : dir.get_files()) {
        _filename = file.full_name();
        cv::Mat _mat = loadIbgrmatWsize(_filename,400,400,false,&isloaded);
        assert(isloaded);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << " - dimensions: " << _mat.cols << "x" << _mat.rows << endl;
        cout << " - depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << " - channels: " << _mat.channels() << endl;
        cv::Mat _tmpmat;
        if(_mat.empty() == false) {
            for(int j = 0; j < 5; ++j) {

                _tmpmat = _mat.clone();

                if(rnd.get_random_float() > 0.5f)
                    cv::flip(_tmpmat,_tmpmat,1);

                _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.05,0.05,5,cv::BORDER_REFLECT,cv::Scalar(0),false);

                _tmpmat = distortimage(_tmpmat,cvrng,0.3);
                /*_tmpmat *= (0.7 + 0.6*rnd.get_random_double());

                int b = rnd.get_integer_in_range(2, 3);
                switch(rnd.get_integer_in_range(0,4)) {
                    case 0:
                        cv::blur(_tmpmat,_tmpmat,cv::Size(b,b));
                        break;
                    case 1:
                        _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),b);
                        _tmpmat = applyMotionBlur(_tmpmat,90.0f*rnd.get_random_float(),b);
                        break;
                    case 2: {
                        int size = b % 2 == 1 ? b : b + 1;
                        cv::GaussianBlur(_tmpmat,_tmpmat,cv::Size(size,size),b);
                    } break;
                    default:
                    break;
                }

                _tmpmat = posterize(_tmpmat,rnd.get_integer_in_range(9,10));

                _tmpmat = addNoise(_tmpmat,cvrng,0,rnd.get_integer_in_range(1,16));

                if(rnd.get_random_float() > 0.5f) {
                    cv::cvtColor(_tmpmat,_tmpmat,cv::COLOR_BGR2GRAY);
                    cv::Mat _chmat[] = {_tmpmat, _tmpmat, _tmpmat};
                    cv::merge(_chmat,3,_tmpmat);
                }


                std::vector<unsigned char> _bytes;
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(static_cast<int>(rnd.get_integer_in_range(15,35)));
                cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
                _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);*/



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


