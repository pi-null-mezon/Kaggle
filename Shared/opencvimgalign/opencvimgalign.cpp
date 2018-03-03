#include "opencvimgalign.h"

cv::Mat alignByPCAWithCrop(const cv::Mat &_inputmat, cv::Size _targetsize, double _thresh, int _interptype, int _bordertype, bool _applyrotation)
{
    if(_inputmat.depth() != CV_8U) {
        CV_Error(cv::Error::BadDepth, "Unsupported image depth");
    }

    // Prepare data for transformation
    cv::Mat _tmpmat, _bwmat;
    switch(_inputmat.channels()) {
        case 4:
            cv::cvtColor(_inputmat,_tmpmat,CV_BGRA2GRAY);
            break;
        case 3:
            cv::cvtColor(_inputmat,_tmpmat,CV_BGR2GRAY);
            break;
        case 1:
            _tmpmat = _inputmat.clone();
            break;
        default:
            CV_Error(cv::Error::BadNumChannels, "Unsupported number of channels");
            break;
    }

    if(_targetsize.area() == 0) {
        _targetsize = cv::Size(_inputmat.cols,_inputmat.rows);
    }
    cv::threshold(_tmpmat,_bwmat,_thresh,255,cv::THRESH_BINARY);

    int _pcalength = (int)(cv::sum(_bwmat)[0]/255);
    if(_pcalength > 0) {
        // Construct a buffer used by the pca analysis
        cv::Mat data_pts = cv::Mat(_pcalength, 2, CV_64FC1);
        unsigned int i = 0;
        for(int y = 0; y < _bwmat.rows; ++y) {
            uchar *pt = _bwmat.ptr<uchar>(y);
            for(int x = 0; x < _bwmat.cols; ++x){
                if(pt[x] == 255) {
                    data_pts.at<double>(i,0) = x;
                    data_pts.at<double>(i,1) = y;
                    i++;
                }
            }
        }

        // Perform PCA analysis
        cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);

        // Store the center of the object
        cv::Point2d _cp(pca_analysis.mean.at<double>(0,0),pca_analysis.mean.at<double>(0,1));
        // Store first eigenvector direction
        cv::Point2d eigenvector(pca_analysis.eigenvectors.at<double>(1,0), pca_analysis.eigenvectors.at<double>(1,1));
        double _angle = 180.0 * std::atan(eigenvector.y / eigenvector.x) / CV_PI;

        // Rotate with the image downscale crop
        cv::Mat _trmat = cv::getRotationMatrix2D(_cp, _angle, 1.0);
        if(_applyrotation) {
            _trmat.at<double>(0,2) += _targetsize.width/2.0 - _cp.x;
            _trmat.at<double>(1,2) += _targetsize.height/2.0 - _cp.y;
        } else {
             _trmat.at<double>(0,0) = 1.0;
             _trmat.at<double>(0,1) = 0.0;
             _trmat.at<double>(0,2) = _targetsize.width/2.0 - _cp.x;

             _trmat.at<double>(1,0) = 0.0;
             _trmat.at<double>(1,1) = 1.0;
             _trmat.at<double>(1,2) = _targetsize.height/2.0 - _cp.y;
        }
        cv::warpAffine(_inputmat, _tmpmat, _trmat, _targetsize, _interptype, _bordertype);
    } else {
        cv::resize(_inputmat,_tmpmat,_targetsize,0,0,_interptype);
    }
    return _tmpmat;
}

cv::Mat projectToPCA(const std::vector<cv::Mat> &_vchannels, unsigned int _targetprojection)
{
    if(_vchannels.size() == 0) {
        CV_Error(cv::Error::BadNumChannels, "Zero size input vector!");
    } else if(_vchannels.size() == 1) {
        return _vchannels[0].clone();
    }
    if(_targetprojection >= _vchannels.size()) {
         CV_Error(cv::Error::BadNumChannels, "Not enough components in the input vector!");
    }

    int _original_cols = _vchannels[0].cols;
    int _original_rows = _vchannels[0].rows;

    std::vector<cv::Mat> _vonerowchannels;
    for(size_t i = 0; i < _vchannels.size(); ++i) {
        if(_vchannels[i].channels() != 1) {
            CV_Error(cv::Error::BadNumChannels, "Unsupported number of channels!");
        }
        _vonerowchannels.push_back(cv::Mat(_vchannels[i]).reshape(0,1));
    }

    cv::Mat _datamat;
    // Concat rows vertically
    cv::vconcat(_vonerowchannels,_datamat);
    // Perform PCA analysis
    cv::PCA pca_analysis(_datamat, cv::Mat(), CV_PCA_DATA_AS_COL);
    cv::Mat _pcaprojmat = pca_analysis.project(_datamat);

    cv::Mat _firstprojmat(_original_rows,_original_cols, CV_32FC1);
    float *_p = _firstprojmat.ptr<float>(0);
    float *_pca = _pcaprojmat.ptr<float>(_targetprojection);
    for(long i = 0; i < _original_cols*_original_rows; ++i) {
        _p[i] = _pca[i];
    }
    return _firstprojmat.reshape(1,_original_rows);
}
