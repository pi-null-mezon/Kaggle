#include <iostream>

#include <opencv2/opencv.hpp>

#include <dlib/data_io.h>
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

    for(auto file : dir.get_files()) {

        cv::Mat _mat = cv::imread(file.full_name(), CV_LOAD_IMAGE_UNCHANGED);
        cout << "---------------------------" << endl;
        cout << "Filename: " << file.full_name() << endl;
        cout << "Img depth (opencv enum 0 - CV_8U, ...): " << _mat.depth() << endl;
        cout << "Img channels: " << _mat.channels() << endl;
        if(_mat.empty() == false) {
            dlib::matrix<dlib::rgb_pixel> _drgbm;
            dlib::array<dlib::matrix<dlib::rgb_pixel>> _vimgs;           
            for(int j = 0; j < 10; ++j) {
                _drgbm = cvmat2dlibmatrix<dlib::rgb_pixel>(_mat);
                dlib::disturb_colors(_drgbm,rnd);                
                //dlib::randomly_crop_image(_drgbm,_vimgs,rnd,1,0.7,1.0,0,0,true,true);
                dlib::randomly_jitter_image(_drgbm,_vimgs,rnd.get_integer(LONG_MAX),1,500,200,1.5,0.11,17.0);
                _drgbm = std::move(_vimgs[0]);
                if(rnd.get_random_float() > 0.5f) {
                    dlib::randomly_cutout_rect(_drgbm,_vimgs,rnd,1,0.5,0.5,180*rnd.get_random_double());
                    _drgbm = std::move(_vimgs[0]);
                }
                for(unsigned long i = 0; i <_vimgs.size(); ++i) {
                    //_vimgs[i] = dlib::fliplr(_vimgs[i]);
                    // Visualise
                    cv::imshow("Probe", dlibmatrix2cvmat<dlib::rgb_pixel>(_drgbm));
                    cv::imshow("Original", _mat);
                    cv::waitKey(0);
                }
            }
        }

    }
    return 0;
}
