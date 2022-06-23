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
cv::Mat cutoutRect(const cv::Mat &_mat, float _cx=0.5f, float _cy=0.5f, float _px=0.5f, float _py=0.5f, float _angledeg=0.0f, bool _meancolor=true, const cv::Scalar &_constcolor=cv::Scalar(104,117,123))
{
    cv::RotatedRect _rrect(cv::Point2f(_mat.cols*_cx,_mat.rows*_cy),cv::Size(_mat.cols*_px,_mat.rows*_py),_angledeg);
    cv::Point2f _verticiesf[4];
    _rrect.points(_verticiesf);
    cv::Point _vert[4];
    for(int i = 0; i < 4; ++i)
        _vert[i] = _verticiesf[i];
    cv::Mat _omat = _mat.clone();
    cv::fillConvexPoly(_omat,_vert,4,_meancolor ? cv::mean(_mat) : _constcolor);
    return _omat;
}

/**
 * @brief Makes image copy with the cutout ellipse on top, color of the cutout region equals to mean color of the input image
 * @param _mat - input image
 * @param _cx - position of the center of the cutout rect  relative to image width
 * @param _cy - position of the center of the cutout rect relative to image width
 * @param _px - portion of the image that will be covered by the cutout rect in horizontal dimension
 * @param _py - portion of the image that will be covered by the cutout rect in vertical dimension
 * @param _angledeg - angle of the cutout rect
 * @return input image copy with the cutout ellipse on top of it
 */
cv::Mat cutoutEllipse(const cv::Mat &_mat, float _cx=0.5f, float _cy=0.5f, float _px=0.5f, float _py=0.5f, float _angledeg=0.0f)
{
    cv::RotatedRect _rrect(cv::Point2f(_mat.cols*_cx,_mat.rows*_cy),cv::Size(_mat.cols*_px,_mat.rows*_py),_angledeg);
    cv::Mat _omat = _mat.clone();
    cv::ellipse(_omat,_rrect,cv::mean(_mat),-1,cv::FILLED);
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
            interpolationMethod = cv::INTER_LINEAR;
        else
            interpolationMethod = cv::INTER_AREA;
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
cv::Mat jitterimage(const cv::Mat &_inmat, cv::RNG &_cvrng, const cv::Size &_targetsize=cv::Size(0,0), double _maxscale=0.05, double _maxshift=0.02, double _maxangle=3, int _bordertype=cv::BORDER_CONSTANT, const cv::Scalar &_constcolor=cv::Scalar(104,117,123), bool _alwaysshrink=false)
{
    cv::Mat _outmat;
    const cv::Size _insize(_inmat.cols,_inmat.rows);
    double _scale = 1.;
    if(_targetsize.area() > 0)
        _scale = std::min((double)_targetsize.width/_insize.width, (double)_targetsize.height/_insize.height);
    cv::Mat _matrix = cv::getRotationMatrix2D(cv::Point2f(_inmat.cols/2.f,_inmat.rows/2.f),
                                              _maxangle * (_cvrng.uniform(0.,2.) - 1.),
                                              _alwaysshrink ? _scale * (1. - _maxscale*_cvrng.uniform(0.,1.0)) : _scale * (1. + _maxscale*(_cvrng.uniform(0.,2.) - 1.)));
    if((_targetsize.width > 0) && (_targetsize.height > 0)) {
        _matrix.at<double>(0,2) += -(_insize.width - _targetsize.width) / 2.;
        _matrix.at<double>(1,2) += -(_insize.height - _targetsize.height) / 2.;
    }
    _matrix.at<double>(0,2) += (_insize.width * _maxshift * _scale * (_cvrng.uniform(0.,2.) - 1.));
    _matrix.at<double>(1,2) += (_insize.height * _maxshift * _scale * (_cvrng.uniform(0.,2.) - 1.));
    cv::warpAffine(_inmat,_outmat,_matrix,
                   _targetsize,
                   _insize.area() > _targetsize.area() ? cv::INTER_AREA : cv::INTER_CUBIC,
                   _bordertype,_constcolor);
    return _outmat;
}

/**
 * @brief distortimage - applies random perspective transformation to input image
 * @param _inmat - input image
 * @param _cvrng - random number generator
 * @return transformed image
 */
cv::Mat distortimage(const cv::Mat&_inmat, cv::RNG &_cvrng, double _maxportion=0.05, int _interp_method=cv::INTER_LINEAR, int _bordertype=cv::BORDER_DEFAULT, const cv::Scalar &_constcolor=cv::Scalar(104,117,123))
{   
    cv::Point2f pts1[]={
                        cv::Point2f(0,0),
                        cv::Point2f(_inmat.cols,0),
                        cv::Point2f(_inmat.cols,_inmat.rows),
                        cv::Point2f(0,_inmat.rows)
                       };
    cv::Point2f pts2[]={
                        cv::Point2f(-_inmat.cols*_cvrng.uniform(-_maxportion,_maxportion),-_inmat.rows*_cvrng.uniform(-_maxportion,_maxportion)),
                        cv::Point2f(_inmat.cols*_cvrng.uniform(1.-_maxportion,1.+_maxportion),-_inmat.rows*_cvrng.uniform(-_maxportion,_maxportion)),
                        cv::Point2f(_inmat.cols*_cvrng.uniform(1.-_maxportion,1.+_maxportion),_inmat.rows*_cvrng.uniform(1.-_maxportion,1.+_maxportion)),
                        cv::Point2f(-_inmat.cols*_cvrng.uniform(-_maxportion,_maxportion),_inmat.rows*_cvrng.uniform(1.-_maxportion,1.+_maxportion))
                       };
    cv::Mat _outmat;
    cv::warpPerspective(_inmat,_outmat,cv::getPerspectiveTransform(pts1,pts2),cv::Size(_inmat.cols,_inmat.rows),_interp_method,_bordertype,_constcolor);
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
        int resizetype = cv::INTER_AREA;
        if(_originalimgmat.cols*_originalimgmat.rows < _tcols*_trows)
            resizetype = cv::INTER_LINEAR;
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
    return _originalimgmat;
}

cv::Mat loadIFbgrmatWsize(std::string _filename, int _tcols, int _trows, bool _crop, bool _center, bool _normalize, bool *_isloadded=nullptr)
{
    cv::Mat _originalimgmat = cv::imread(_filename, cv::IMREAD_COLOR);
    if(_isloadded)
        *_isloadded = !_originalimgmat.empty();

    if(_originalimgmat.empty())
        return cv::Mat();

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
            const float _mean_blue  = static_cast<float>(_vchannelmean.at<const double>(0));
            const float _mean_green = static_cast<float>(_vchannelmean.at<const double>(1));
            const float _mean_red   = static_cast<float>(_vchannelmean.at<const double>(2));
            const float _stdev_blue  = static_cast<float>(3.0*_vchannelstdev.at<const double>(0));
            const float _stdev_green = static_cast<float>(3.0*_vchannelstdev.at<const double>(1));
            const float _stdev_red   = static_cast<float>(3.0*_vchannelstdev.at<const double>(2));
            float *_val = _originalimgmat.ptr<float>(0);
            size_t _pos = 0;
            for(size_t i = 0; i < _originalimgmat.total(); ++i) {
                _pos = i*3;
                _val[_pos]   = (_val[_pos]   - _mean_blue)  / _stdev_blue;
                _val[_pos+1] = (_val[_pos+1] - _mean_green) / _stdev_green;
                _val[_pos+2] = (_val[_pos+2] - _mean_red)   / _stdev_red;
            }
        } else {
            cv::Scalar _vchannelmean = cv::mean(_originalimgmat);
            _originalimgmat = (_originalimgmat - _vchannelmean) / 256.0;
        }
    }
    return _originalimgmat;
}

cv::Mat addNoise(const cv::Mat &_inmat, cv::RNG &_cvrng, double _a=0, double _b=0.1, int _distributiontype=cv::RNG::NORMAL)
{
    cv::Mat _tmpmat = cv::Mat::zeros(_inmat.rows,_inmat.cols,_inmat.type());
    _cvrng.fill(_tmpmat,_distributiontype,_a,_b);
    return _inmat + _tmpmat;
}

cv::Mat applyMotionBlur(const cv::Mat &_inmat, float angle, int size)
{
    cv::Mat kernel = cv::Mat::zeros(size,size,CV_32FC1);
    float *p = kernel.ptr<float>(size/2);
    for(int i = 0; i < size; ++i)
        p[i] = 1.0f / size;
    cv::warpAffine(kernel,kernel,cv::getRotationMatrix2D(cv::Point2f(size/2.0f,size/2.0f),angle,1),cv::Size(size,size),cv::INTER_CUBIC);
    cv::Mat result;
    cv::filter2D(_inmat,result,_inmat.depth(),kernel);
    return result;
}

cv::Mat applyFlare(const cv::Mat &_inmat, cv::RNG &_cvrng, float hpos=0.5f, float vpos = 0.5f)
{
    cv::Mat mask = cv::Mat::zeros(_inmat.cols,_inmat.rows,_inmat.type());
    cv::Point2f sp(_inmat.cols*hpos,_inmat.rows*vpos);
    cv::Point2f cp(mask.cols/2.0f,mask.rows/2.0f);
    int steps = _cvrng.uniform(5,40);
    float step = mask.rows / steps;
    for(int i = 0; i < steps; ++i) {
        cv::Point2f cp(mask.cols/2.0f,mask.rows - step*i);
        cv::Point2f diff = (cp - sp);
        cv::line(mask,sp,sp + _cvrng.uniform(0.15f,2.25f)*diff,
                 cv::Scalar(155 + 100*_cvrng.uniform(0.0f,1.0f),
                            155 + 100*_cvrng.uniform(0.0f,1.0f),
                            155 + 100*_cvrng.uniform(0.0f,1.0f)),1,cv::LINE_AA);
        if(_cvrng.uniform(0.0f,1.0f) > 0.9f)
            cv::circle(mask,sp + _cvrng.uniform(0.15f,2.25f)*diff,
                       _cvrng.uniform(mask.cols/40,mask.cols/2),
                       cv::Scalar(155 + 100*_cvrng.uniform(0.0f,1.0f),
                                  155 + 100*_cvrng.uniform(0.0f,1.0f),
                                  155 + 100*_cvrng.uniform(0.0f,1.0f)),
                       1,cv::LINE_AA);

    }
    cv::blur(mask,mask,cv::Size(_cvrng.uniform(5,43),_cvrng.uniform(5,43)));
    return _inmat + mask;
}

cv::Mat posterize(const cv::Mat &bgrmat, uint8_t lvls)
{
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    float step = 255.0f / lvls;
    for(int i = 0; i < 256; ++i)
        p[i] = static_cast<uchar>(step * std::floor(i / step));
    cv::Mat omat;
    cv::LUT(bgrmat,lookUpTable,omat);
    return omat;
}

#endif // OPENCVIMGAUGMENT_H
