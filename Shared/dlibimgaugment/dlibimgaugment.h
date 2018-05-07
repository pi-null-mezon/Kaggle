/*
Commonly used image randomization facilities are stored here
*/
#ifndef DLIBIMGAUGMENT_H
#define DLIBIMGAUGMENT_H

#include <dlib/image_transforms.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"

namespace dlib {

// Do not forget about existance of dlib::apply_random_color_offset(crop, rnd)

template<typename image_type>
void randomly_jitter_image(const matrix<image_type>& img, dlib::array<matrix<image_type>>& crops, time_t seed, size_t num_crops, unsigned long _trows=0, unsigned long _tcols=0, double _maxsizepercents=1.1, double _maxtranslationpercents=0.05, double _maxrotationdeg=15.0)
{
    if(_tcols == 0)
        _tcols = num_columns(img);
    if(_trows == 0)
        _trows = num_rows(img);

    thread_local random_cropper cropper; // so the jitter_images could be used in the threads with the minimum cost (one random_cropper per thread will be created)
    cropper.set_seed(seed);
    cropper.set_chip_dims(_trows,_tcols);
    cropper.set_randomly_flip(false);
    cropper.set_background_crops_fraction(0);
    cropper.set_max_object_size(_maxsizepercents);
    cropper.set_min_object_size(std::max(_trows,_tcols),std::min(_trows,_tcols));
    cropper.set_translate_amount(_maxtranslationpercents);
    cropper.set_max_rotation_degrees(_maxrotationdeg);

    dlib::rectangle _imgrect = get_rect(img);

    std::vector<mmod_rect> _boxes(num_crops, shrink_rect(_imgrect,(unsigned long)(_imgrect.width()*0.025))), _crop_boxes;
    matrix<image_type> _tmpimg;
    crops.resize(num_crops);
    for(size_t i = 0; i < num_crops; ++i) {       
        cropper(img, _boxes, _tmpimg, _crop_boxes);
        crops[i] = _tmpimg;
    }
}

template<typename image_type>
rectangle make_random_cropping_rect(const matrix<image_type> &img, dlib::rand &rnd, float mins=0.900f, float maxs=0.999f)
{   
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    rectangle rect(scale*img.nc(), scale*img.nr());
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

template<typename image_type>
void randomly_crop_image(const matrix<image_type>& img,dlib::array<matrix<image_type>>& crops, dlib::rand& rnd, long num_crops, float _mins, float _maxs, unsigned long _trows=0, unsigned long _tcols=0)
{
    if(_tcols == 0)
        _tcols = num_columns(img);
    if(_trows == 0)
        _trows = num_rows(img);

    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i) {
        auto rect = make_random_cropping_rect(img, rnd, _mins, _maxs);
        dets.push_back(chip_details(rect, chip_dims(_trows,_tcols)));
    }
    extract_image_chips(img, dets, crops);
}

template<typename image_type>
void randomly_cutout_rect(const matrix<image_type>& img, dlib::array<matrix<image_type>>& crops, dlib::rand& rnd, long num_crops, float _px=0.5f, float _py=0.5f, float _angledeg=0.0)
{   
    cv::Mat _tmpmatimg = dlibmatrix2cvmat<image_type>(img);
    crops.resize(num_crops);
    for(unsigned long i = 0; i < crops.size(); ++i) {
        cv::Mat _tmpmat = cutoutRect(_tmpmatimg,rnd.get_random_float(),rnd.get_random_float(),_px,_py,_angledeg);
        crops[i] = cvmat2dlibmatrix<image_type>(_tmpmat);
    }
}


dlib::matrix<dlib::rgb_pixel> load_rgb_image_with_fixed_size(std::string _filename, int _trows, int _tcols, bool _crop, bool *_isloadded=0)
{
    cv::Mat _originalimgmat = cv::imread(_filename, CV_LOAD_IMAGE_COLOR);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return dlib::matrix<dlib::rgb_pixel>();

    if(_crop == true)
        return cvmat2dlibmatrix<dlib::rgb_pixel>(cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows)));

    if(_originalimgmat.cols > _tcols || _originalimgmat.rows > _trows)
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,CV_INTER_AREA);
    else if(_originalimgmat.cols < _tcols || _originalimgmat.rows < _trows)
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,CV_INTER_LINEAR);
    return cvmat2dlibmatrix<dlib::rgb_pixel>(_originalimgmat);
}

} // end of dlib namespace

#endif // DLIBIMGAUGMENT_H
