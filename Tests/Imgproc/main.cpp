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

        cv::Mat _mat = cv::imread(file.full_name(), CV_LOAD_IMAGE_COLOR);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        if(_mat.empty() == false) {

            //_mat = std::move(distortimage(_mat,cvrng));

            /*for(int j = 0; j < 5; ++j) {
                cv::Mat _tmpmat = std::move(jitterimage(_mat,cvrng));
                //_tmpmat = std::move(distortimage(_tmpmat,cvrng));
                cv::imshow("Probe", _tmpmat);
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }*/


            for(int j = 0; j < 10; ++j) {
                cv::resize(_mat,_mat,cv::Size(250,150),0,0,cv::INTER_AREA);
                dlib::matrix<dlib::rgb_pixel> _drgbm = std::move( cvmat2dlibmatrix<dlib::rgb_pixel>(_mat) );
                dlib::array<dlib::matrix<dlib::rgb_pixel>> _vimgs;

                //dlib::randomly_crop_image(_drgbm,_vimgs,rnd,1,0.900,0.999,250,150,false,true);
                //_drgbm = std::move(_vimgs[0]);
                if(rnd.get_random_float() > 0.0f) {
                   dlib::randomly_jitter_image(_drgbm,_vimgs,rnd.get_integer(LONG_MAX),1,0,0,1.0,0.02,10);
                    _drgbm = std::move(_vimgs[0]);
                    dlib::disturb_colors(_drgbm,rnd);
                }
                /*if(rnd.get_random_float() > 0.1f) {
                    dlib::randomly_cutout_rect(_drgbm,_vimgs,rnd,1,0.1,0.1);
                    _drgbm = std::move(_vimgs[0]);
                }*/
                cv::imshow("Probe", dlibmatrix2cvmat<dlib::rgb_pixel>(_drgbm));
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }
        }

    }
    return 0;
}
