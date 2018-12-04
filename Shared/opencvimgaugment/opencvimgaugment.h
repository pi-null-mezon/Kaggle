#ifndef OPENCVIMGAUGMENT_H
#define OPENCVIMGAUGMENT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


/**
 * @brief Makes image copy with the cutout rectangle on top, color of the cutout region equals to mean color of the input image
 * @param _mat - input image
 * @param _cx - position of the center of the cutout rect  relative to image width
 * @param _cy - position of the center of the cutout rect relative to image width
 * @param _px - portion of the image that will be covered by the cutout rect in horizontal dimension
 * @param _py - portion of the image that will be covered by the cutout rect in vertical dimension
 * @param _angledeg - angle of the cutout rect
 * @return input image copy with the cutout rect on top of it
 */
cv::Mat cutoutRect(const cv::Mat &_mat, float _cx=0.5f, float _cy=0.5f, float _px=0.5f, float _py=0.5f, float _angledeg=0.0f)
{
    cv::RotatedRect _rrect(cv::Point2f(_mat.cols*_cx,_mat.rows*_cy),cv::Size(_mat.cols*_px,_mat.rows*_py),_angledeg);
    cv::Point2f _verticiesf[4];
    _rrect.points(_verticiesf);
    cv::Point _vert[4];
    for(int i = 0; i < 4; ++i)
        _vert[i] = _verticiesf[i];
    cv::Mat _omat = _mat.clone();
    cv::fillConvexPoly(_omat,_vert,4,cv::mean(_mat));
    return _omat;
}

/**
 * @brief Makes fixed size random (if _rng is not nullptr, if null crops from center) crop of the input image
 * @param _inmat - input image
 * @param _cropsize - crop size (should be less than input image size)
 * @param _rng - random number generator
 * @return random crop with fixed size
 */
cv::Mat cropimage(const cv::Mat &_inmat, const cv::Size &_cropsize, cv::RNG *_rng=nullptr)
{
    int _dw = _inmat.cols - _cropsize.width;
    int _dh = _inmat.rows - _cropsize.height;
    cv::Rect _rect(cv::Point(_dw/2,_dh/2),_cropsize);
    if(_rng != nullptr)
        _rect = cv::Rect(cv::Point(_dw*_rng->uniform(0.f,1.f),_dh*_rng->uniform(0.f,1.f)),_cropsize);
    return cv::Mat(_inmat,_rect).clone();
}


/**
 * @brief Produces random crops of the input image. All crops represents fixed sizes regions of the original image
 * @param _inmat - input image
 * @param _cropsnum - number of the crops
 * @param _vmats - vector of the output crops
 * @param _rng - reference to random number generator
 * @param _cropsize - target size of the crops, it also represents the size of the part of the original image tha should be cropped
 */
void getImageFSCrops(const cv::Mat &_inmat, const size_t _cropsnum, std::vector<cv::Mat> &_vmats, cv::RNG &_rng, const cv::Size &_cropsize)
{
    _vmats.resize(_cropsnum);
    int _dw = _inmat.cols - _cropsize.width;
    int _dh = _inmat.rows - _cropsize.height;
    for(size_t i = 0; i < _vmats.size(); ++i) {
        cv::Rect _rect(cv::Point(_dw*_rng.uniform(0.f,1.f),_dh*_rng.uniform(0.f,1.f)),_cropsize);
        _vmats[i] = cv::Mat(_inmat,_rect).clone();
    }
}

/**
 * @brief Makes crop with defined size from the image center (first makes crop with defined size proportion only then makes resize)
 * @param input - self explained
 * @param size - target size
 * @return cropped patch of the image
 */
cv::Mat cropFromCenterAndResize(const cv::Mat &input, cv::Size size)
{
    cv::Rect2f roiRect(0,0,0,0);
    if( (float)input.cols/input.rows > (float)size.width/size.height) {
        roiRect.height = (float)input.rows;
        roiRect.width = input.rows * (float)size.width/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = (float)input.cols;
        roiRect.height = input.cols * (float)size.height/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0, 0, (float)input.cols, (float)input.rows);
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = 0;
        if(size.area() > roiRect.area())
            interpolationMethod = CV_INTER_LINEAR;
        else
            interpolationMethod = CV_INTER_AREA;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}

/**
 * @brief Apply random affine transformation to image
 * @param _inmat - input image
 * @param _cvrng - random number generator
 * @param _targetsize - target output image size
 * @param _maxscale - scale deviation
 * @param _maxshift - translation deviation
 * @param _maxangle - angle deviation in degrees
 * @param _bordertype - opencv border type
 * @return transformed image
 */
cv::Mat jitterimage(const cv::Mat &_inmat, cv::RNG &_cvrng, const cv::Size &_targetsize=cv::Size(0,0), double _maxscale=0.05, double _maxshift=0.02, double _maxangle=3, int _bordertype=cv::BORDER_CONSTANT)
{
    cv::Mat _outmat;
    const cv::Size _insize(_inmat.cols,_inmat.rows);
    double _scale = 1.;
    if(_targetsize.area() > 0)
        _scale = std::min((double)_targetsize.width/_insize.width, (double)_targetsize.height/_insize.height);
    cv::Mat _matrix = cv::getRotationMatrix2D(cv::Point2f(_inmat.cols/2.f,_inmat.rows/2.f),
                                              _maxangle * (_cvrng.uniform(0.,2.) - 1.),
                                              _scale * (1. + _maxscale*(_cvrng.uniform(0.,2.) - 1.)));
    if((_targetsize.width > 0) && (_targetsize.height > 0)) {
        _matrix.at<double>(0,2) += -(_insize.width - _targetsize.width) / 2.;
        _matrix.at<double>(1,2) += -(_insize.height - _targetsize.height) / 2.;
    }
    _matrix.at<double>(0,2) += (_insize.width * _maxshift * _scale * (_cvrng.uniform(0.,2.) - 1.));
    _matrix.at<double>(1,2) += (_insize.height * _maxshift * _scale * (_cvrng.uniform(0.,2.) - 1.));
    cv::warpAffine(_inmat,_outmat,_matrix,
                   _targetsize,
                   _insize.area() > _targetsize.area() ? CV_INTER_AREA : CV_INTER_CUBIC,
                   _bordertype,cv::Scalar(104,117,123));
    return _outmat;
}

/**
 * @brief distortimage - applies random perspective transformation to input image
 * @param _inmat - input image
 * @param _cvrng - random number generator
 * @return transformed image
 */
cv::Mat distortimage(const cv::Mat&_inmat, cv::RNG &_cvrng, double _maxportion=0.05, int _interp_method=CV_INTER_LINEAR, int _bordertype=cv::BORDER_DEFAULT)
{   
    cv::Point2f pts1[]={cv::Point2f(0,0),
    cv::Point2f(_inmat.cols,0),
    cv::Point2f(_inmat.cols,_inmat.rows),
    cv::Point2f(0,_inmat.rows)};
    cv::Point2f pts2[]={cv::Point2f(-_inmat.cols*_cvrng.uniform(-_maxportion,_maxportion),-_inmat.rows*_cvrng.uniform(-_maxportion,_maxportion)),
    cv::Point2f(_inmat.cols*_cvrng.uniform(1.-_maxportion,1.+_maxportion),-_inmat.rows*_cvrng.uniform(-_maxportion,_maxportion)),
    cv::Point2f(_inmat.cols*_cvrng.uniform(1.-_maxportion,1.+_maxportion),_inmat.rows*_cvrng.uniform(1.-_maxportion,1.+_maxportion)),
    cv::Point2f(-_inmat.cols*_cvrng.uniform(-_maxportion,_maxportion),_inmat.rows*_cvrng.uniform(1.-_maxportion,1.+_maxportion))};
    cv::Mat _outmat;
    cv::warpPerspective(_inmat,_outmat,cv::getPerspectiveTransform(pts1,pts2),cv::Size(_inmat.cols,_inmat.rows),_interp_method,_bordertype,cv::Scalar(104,117,123));
    return _outmat;
}

cv::Mat loadIbgrmatWsize(std::string _filename, int _tcols, int _trows, bool _crop, bool *_isloadded=nullptr)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_COLOR);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return cv::Mat();

    if(_crop == true)
        return cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows));

    if((_originalimgmat.cols != _tcols) || (_originalimgmat.rows != _trows)) {
        int resizetype = CV_INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = CV_INTER_LINEAR;
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,resizetype);
    }
    return _originalimgmat;
}

cv::Mat loadIFgraymatWsize(std::string _filename, int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded=nullptr)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_GRAYSCALE);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return cv::Mat();

    if(_crop == true) {
        _originalimgmat = cropFromCenterAndResize(_originalimgmat,cv::Size(_tcols,_trows));
    } else if((_originalimgmat.cols != _tcols) || (_originalimgmat.rows != _trows)) {
        int resizetype = CV_INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = CV_INTER_CUBIC;
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
    return _originalimgmat;
}

#endif // OPENCVIMGAUGMENT_H
