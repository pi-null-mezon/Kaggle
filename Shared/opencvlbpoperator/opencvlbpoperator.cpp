#include "opencvlbpoperator.h"

namespace cv { namespace lbph {

double compareLBPH(const Mat &_refmat, const Mat &_testmat, int _radius, int _neighbors, int _grid_x, int _grid_y, bool _normed, bool _extended)
{
    if(_refmat.rows != _testmat.rows || _refmat.cols != _refmat.cols) {
        String error_msg = "Not equal sizes of compared images. Please pass the images with equal sizes!";
        CV_Error(Error::StsNotImplemented, error_msg);
    }
    Mat lbp_refimage = _extended ? elbp(_refmat, _radius, _neighbors) : olbp(_refmat);
    Mat _refhist = spatial_histogram(
            lbp_refimage,
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))),
            _grid_x,
            _grid_y,
            _normed);

    Mat lbp_testimage = _extended ? elbp(_testmat, _radius, _neighbors) : olbp(_testmat);
    Mat _testhist = spatial_histogram(
            lbp_testimage,
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))),
            _grid_x,
            _grid_y,
            _normed);
    return norm(_refhist, _testhist, NORM_L2);
}

cv::Mat getELBPHistogramm(const cv::Mat &_img, int _radius, int _neighbors, int _grid_x, int _grid_y, bool _normed)
{
    Mat lbp_image = elbp(_img, _radius, _neighbors);
    return spatial_histogram(
            lbp_image,
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))),
            _grid_x,
            _grid_y,
            _normed);
}

cv::Mat getOLBPHistogramm(const cv::Mat &_img, int _grid_x, int _grid_y, bool _normed)
{
    Mat lbp_image = olbp(_img);
    return spatial_histogram( lbp_image, 256, _grid_x, _grid_y, _normed);
}

}}
