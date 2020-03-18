#ifndef DLIBOPENCVCONVERTER_H
#define DLIBOPENCVCONVERTER_H

#include <assert.h>

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
    assert(_cvmat.channels() == 3);
    assert(_cvmat.depth() == CV_8U);
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_cvmat.rows,_cvmat.cols);
    for(long i = 0; i < _cvmat.rows*_cvmat.cols; ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;

    // Alternative way (works little bit slower)
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
    assert(_cvmat.channels() == 1);
    assert(_cvmat.depth() == CV_32F);
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    float *_p = _mat.ptr<float>(0);
    dlib::matrix<float> _img(_cvmat.rows,_cvmat.cols);
    for(long i = 0; i < _cvmat.cols*_cvmat.rows; ++i)
        _img(i) = _p[i];
    return _img;
}

// function template specialization
template <long C>
std::array<dlib::matrix<float>,C> cvmatF2arrayofFdlibmatrix(const cv::Mat &_cvmat)
{
    assert(_cvmat.channels() == C);
    assert(_cvmat.depth() == CV_32F);
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();

    std::array<dlib::matrix<float>,C> _img;
    for(size_t _channel = 0; _channel < _img.size(); ++_channel) {
        float *_p = _mat.ptr<float>(0);
        dlib::matrix<float> _chimg(_cvmat.rows,_cvmat.cols);
        for(long i = 0; i < _mat.rows*_mat.cols; ++i) {
            _chimg(i) = _p[C*i+_channel];
        }
        _img[_channel] = std::move(_chimg);
    }
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
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_8UC3, const_cast<void*>(dlib::image_data(_img)));
    cv::Mat _outmat;
    cv::cvtColor(_tmpmat,_outmat,cv::COLOR_BGR2RGB);
    return _outmat;
}

// function template specialization
template <>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<uchar> &_img)
{
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_8UC1, const_cast<void*>(dlib::image_data(_img)));
    return _tmpmat.clone();
}

// function template specialization
template <>
cv::Mat dlibmatrix2cvmat(const dlib::matrix<float> &_img)
{
    cv::Mat _tmpmat(dlib::num_rows(_img), dlib::num_columns(_img), CV_32FC1, const_cast<void*>(dlib::image_data(_img)));
    return _tmpmat.clone();
}


#endif // DLIBOPENCVCONVERTER_H
