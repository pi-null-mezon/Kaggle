/*
Commonly used image randomization facilities are stored here
*/
#ifndef DLIBIMGAUGMENT_H
#define DLIBIMGAUGMENT_H

#include <dlib/image_transforms.h>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"

namespace dlib {

// Do not forget about existance of dlib::apply_random_color_offset(crop, rnd)

template<typename image_type>
void randomly_jitter_image(const matrix<image_type>& img, dlib::array<matrix<image_type>>& crops, time_t seed, size_t num_crops, unsigned long _trows=0, unsigned long _tcols=0)
{
    if(_tcols == 0)
        _tcols = num_columns(img);
    if(_trows == 0)
        _trows = num_rows(img);

    thread_local random_cropper cropper; // so the jitter_images could be used in the threads with the minimum cost (one random_cropper per thread will be created)
    cropper.set_seed(time(0)+seed);
    cropper.set_chip_dims(_trows,_tcols);
    cropper.set_randomly_flip(false);
    cropper.set_background_crops_fraction(0);
    cropper.set_max_object_size(1.07);
    cropper.set_min_object_size(1.01);
    cropper.set_translate_amount(0.0085);
    cropper.set_max_rotation_degrees(0.85);

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
rectangle make_random_cropping_rect(const matrix<image_type> &img, dlib::rand &rnd)
{
    // figure out what rectangle we want to crop from the image
    double mins = 0.91, maxs = 0.99; // do not make greater than 1.0 or app will silently crash
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    auto size = scale*std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

template<typename image_type>
void randomly_crop_image(const matrix<image_type>& img,dlib::array<matrix<image_type>>& crops, dlib::rand& rnd, long num_crops, unsigned long _trows=0, unsigned long _tcols=0)
{
    if(_tcols == 0)
        _tcols = num_columns(img);
    if(_trows == 0)
        _trows = num_rows(img);

    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i) {
        auto rect = make_random_cropping_rect(img, rnd);
        dets.push_back(chip_details(rect, chip_dims(_trows,_tcols)));
    }
    extract_image_chips(img, dets, crops);

    /*for (auto&& img : crops) {
        // Also randomly flip the image
        if(rnd.get_random_double() < 0.5)
            img = fliplr(img);
        if(rnd.get_random_double() > 0.5)
            img = flipud(img);
    }*/
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


} // end of dlib namespace

#endif // DLIBIMGAUGMENT_H
