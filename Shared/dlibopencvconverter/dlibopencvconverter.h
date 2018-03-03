#ifndef DLIBOPENCVCONVERTER_H
#define DLIBOPENCVCONVERTER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing.h>

using namespace std;

// --------------------------------------- DLIB TO OPENCV ---------------------------------------

// function template
template <typename image_type>
dlib::matrix<image_type> cvmat2dlibmatrix(const cv::Mat &_cvmat);

// function template specialization
template <>
dlib::matrix<dlib::rgb_pixel> cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    // TO DO checks of the channels number and color depth
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_cvmat.rows,_cvmat.cols);
    for(long i = 0; i < _cvmat.rows*_cvmat.cols; ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;

    // Alternative way
    /*dlib::matrix<dlib::rgb_pixel> _om;
    dlib::cv_image<dlib::rgb_pixel> _iimg(_cvmat);
    dlib::assign_image(_om,_iimg);
    return _om;*/
}

// function template specialization
template <>
dlib::matrix<uchar> cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    // TO DO checks of the channels number and color depth
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<uchar> _img(_cvmat.rows,_cvmat.cols);
    for(long i = 0; i < _cvmat.cols*_cvmat.rows; ++i)
        _img(i) = _p[i];
    return _img;
}

// function template specialization
template <>
dlib::matrix<float> cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    // TO DO checks of the channels number and color depth
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    float *_p = _mat.ptr<float>(0);
    dlib::matrix<float> _img(_cvmat.rows,_cvmat.cols);
    for(long i = 0; i < _cvmat.cols*_cvmat.rows; ++i)
        _img(i) = _p[i];
    return _img;
}

// --------------------------------------- OPENCV TO DLIB ---------------------------------------

// function template
template <typename image_type>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<image_type> &_img);

// function template specialization
template <>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<dlib::rgb_pixel> &_img)
{   
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_8UC3, (void*)dlib::image_data(_img));
    cv::cvtColor(_tmpmat,_tmpmat,CV_RGB2BGR);
    return _tmpmat;
}

// function template specialization
template <>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<uchar> &_img)
{
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_8UC1, (void*)dlib::image_data(_img));
    return _tmpmat.clone();
}

// function template specialization
template <>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<float> &_img)
{
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_32FC1, (void*)dlib::image_data(_img));
    return _tmpmat.clone();
}


#endif // DLIBOPENCVCONVERTER_H
