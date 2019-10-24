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
void randomly_jitter_image(const matrix<image_type>& img, dlib::array<matrix<image_type>>& crops, time_t seed, size_t num_crops, unsigned long _tcols=0, unsigned long _trows=0, double _maxsizepercents=1.1, double _maxtranslationpercents=0.05, double _maxrotationdeg=15.0)
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

    std::vector<mmod_rect> _boxes(num_crops, shrink_rect(_imgrect,(unsigned long)(_imgrect.width()*0.01))), _crop_boxes;
    matrix<image_type> _tmpimg;
    crops.resize(num_crops);
    for(size_t i = 0; i < num_crops; ++i) {       
        cropper(img, _boxes, _tmpimg, _crop_boxes);
        crops[i] = _tmpimg;
    }
}

template<typename image_type>
rectangle make_random_cropping_rect(const matrix<image_type> &img, dlib::rand &rnd, float _mins, float _maxs, long _tcols, long _trows)
{   
    auto _scale = _mins + rnd.get_random_double()*(_maxs-_mins);
    _tcols *= _scale;
    _trows *= _scale;
    assert(_tcols < img.nc());
    assert(_trows < img.nr());
    rectangle rect(_tcols, _trows);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

template<typename image_type>
void randomly_crop_image(const matrix<image_type>& img,dlib::array<matrix<image_type>>& crops, dlib::rand& rnd, long num_crops, float _mins, float _maxs, unsigned long _tcols=0, unsigned long _trows=0, bool _rndfliplr=false, bool _rnddisturbcolors=false)
{
    if(_tcols == 0)
        _tcols = num_columns(img);
    if(_trows == 0)
        _trows = num_rows(img);

    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i) {
        auto rect = make_random_cropping_rect(img, rnd, _mins, _maxs,_tcols,_trows);
        dets.push_back(chip_details(rect, chip_dims(_trows,_tcols)));
    }
    extract_image_chips(img, dets, crops);
    for(auto&& _tmpimg : crops)
    {
        // Also randomly flip the image
        if(_rndfliplr) {
            if (rnd.get_random_double() > 0.5)
                _tmpimg = fliplr(_tmpimg);
        }

        // And then randomly adjust the colors.
        if(_rnddisturbcolors)
            disturb_colors(_tmpimg, rnd);
        //apply_random_color_offset(_tmpimg, rnd);
    }

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

dlib::matrix<dlib::rgb_pixel> load_rgb_image_with_fixed_size(std::string _filename, int _tcols, int _trows, bool _crop, bool *_isloadded=0)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_COLOR);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return dlib::matrix<dlib::rgb_pixel>();

    if(_crop == true)
        return cvmat2dlibmatrix<dlib::rgb_pixel>(cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows)));

    if((_originalimgmat.cols != _tcols) || (_originalimgmat.rows != _trows)) {
        int resizetype = cv::INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = cv::INTER_LINEAR;
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,resizetype);
    }
    return cvmat2dlibmatrix<dlib::rgb_pixel>(_originalimgmat);
}

dlib::matrix<uchar> load_grayscale_image_with_fixed_size(std::string _filename, int _tcols, int _trows, bool _crop, bool *_isloadded=0)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_GRAYSCALE);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return dlib::matrix<uchar>();

    if(_crop == true)
        return cvmat2dlibmatrix<uchar>(cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows)));

    if((_originalimgmat.cols != _tcols) || (_originalimgmat.rows != _trows)) {
        int resizetype = cv::INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = cv::INTER_CUBIC;
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,resizetype);
    }
    return cvmat2dlibmatrix<uchar>(_originalimgmat);
}

dlib::matrix<float> load_grayscale_image_with_fixed_size(const std::string& _filename, int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded=0)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_GRAYSCALE);

    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return dlib::matrix<float>();

    if(_crop == true) {
        _originalimgmat = cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows));
    } else if((_originalimgmat.cols != _tcols) || (_originalimgmat.rows != _trows)) {
        int resizetype = cv::INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = cv::INTER_CUBIC;
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,resizetype);
    }

    _originalimgmat.convertTo(_originalimgmat,CV_32F);
    if(_center) {
        if(_normalize) {
            cv::Mat _vchannelmean, _vchannelstdev;
            cv::meanStdDev(_originalimgmat,_vchannelmean,_vchannelstdev);
            _originalimgmat = (_originalimgmat - _vchannelmean.at<const double>(0)) / (3.0*_vchannelstdev.at<const double>(0));
        } else {
            cv::Scalar _mean = cv::mean(_originalimgmat);
            _originalimgmat = (_originalimgmat - _mean[0]) / 256;
        }
    }
    return cvmat2dlibmatrix<float>(_originalimgmat);
}

} // end of dlib namespace

#endif // DLIBIMGAUGMENT_H
