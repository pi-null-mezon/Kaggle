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

            for(int j = 0; j < 5; ++j) {
                cv::Mat _tmpmat = std::move(jitterimage(_mat,cvrng));
                //_tmpmat = std::move(distortimage(_tmpmat,cvrng));
                cv::imshow("Probe", _tmpmat);
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }


            /*for(int j = 0; j < 10; ++j) {
                cv::resize(_mat,_mat,cv::Size(500,200),0,0,CV_INTER_CUBIC);
                dlib::matrix<dlib::rgb_pixel> _drgbm = std::move( cvmat2dlibmatrix<dlib::rgb_pixel>(_mat) );

                dlib::apply_random_color_offset(_drgbm,rnd);
                dlib::array<dlib::matrix<dlib::rgb_pixel>> _vimgs;
                if(rnd.get_random_float() > 0.2f) {
                    dlib::randomly_jitter_image(_drgbm,_vimgs,rnd.get_integer(LONG_MAX),1,0,0,1.11,0.05,13.0);
                    _drgbm = std::move(_vimgs[0]);
                }
                if(rnd.get_random_float() > 0.2f) {
                    dlib::randomly_crop_image(_drgbm,_vimgs,rnd,1,0.800,0.999,0,0,true);
                    _drgbm = std::move(_vimgs[0]);
                }
                if(rnd.get_random_float() > 0.2f) {
                    dlib::randomly_cutout_rect(_drgbm,_vimgs,rnd,1,0.5,0.5);
                    _drgbm = std::move(_vimgs[0]);
                }
                cv::imshow("Probe", dlibmatrix2cvmat<dlib::rgb_pixel>(_drgbm));
                cv::imshow("Original", _mat);
                cv::waitKey(0);
            }*/
        }

    }
    return 0;
}
