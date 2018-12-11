#ifndef OPENCVMORPH_H
#define OPENCVMORPH_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief openAreaAndBorders - makes morphological opening of the connected components with area smaller tahn threshold or touching  image border
 * @param _boolimg - input binary image of CV_8UC1 type
 * @param _labeledimg - output binary image with CV_8UC1 type
 * @param connectivity - self explained (8 or 4);
 * @param _areathreshold - minimum area of the components that should be preserved
 * @param _openborders - controls should be components adjusted to image border removed or not
 */
void openAreaAndBorders(cv::InputArray _boolimg, cv::OutputArray _labeledimg, int connectivity=8, unsigned int _areathreshold=0, bool _openborders=true)
{
    cv::Mat _lmat;
    int _totalcomp = cv::connectedComponents(_boolimg, _lmat, connectivity, CV_32S);
    std::vector<unsigned int> _varea(_totalcomp, 0);
    std::vector<bool> _vbordertouch(_totalcomp,false);
    // Let's count area of each connected component
    int *p = nullptr;
    for(int y = 0; y < _lmat.rows; y++) {
        p = _lmat.ptr<int>(y);
        for(int x = 0; x < _lmat.cols; x++) {
            _varea[p[x]] += 1;
            if(x == 0 || x == _lmat.cols-1 || y == 0 || y == _lmat.rows-1)
                _vbordertouch[p[x]] = true;
        }
    }
    cv::Mat _outmat = cv::Mat::zeros(_lmat.rows, _lmat.cols, CV_8UC1);
    unsigned char *ptrout;
    for(int y = 0; y < _lmat.rows; y++) {
        p = _lmat.ptr<int>(y);
        ptrout = _outmat.ptr(y);
        if(_openborders) {
            for(int x = 0; x < _lmat.cols; x++)
                if(_varea[p[x]] > _areathreshold && p[x] != 0 && _vbordertouch[p[x]] == false)
                    ptrout[x] = 255;
        } else {
            for(int x = 0; x < _lmat.cols; x++)
                if(_varea[p[x]] > _areathreshold && p[x] != 0)
                    ptrout[x] = 255;
        }
    }
    _labeledimg.assign(_outmat);
}

/**
 * @brief skeleton computes skeleton of the connected components
 * @param _boolimg - input binary image of CV_8UC1 type
 * @param _labeledimg - output binary image with CV_8UC1 type
 */
void skeleton(cv::InputArray _boolimg, cv::OutputArray _labeledimg)
{
    cv::Mat img = _boolimg.getMat();
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp, eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    bool done = false;
    do {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);
        done = (cv::countNonZero(img) == 0);
    } while (!done);
    _labeledimg.assign(skel);
}

/**
 * @brief extracts external border of connected components
 * @param _binaryimg - input binary image of CV_8UC1 type
 * @param _outbinaryimg - output binary image with CV_8UC1 type
 * @param _elsize -  size of morph element
 */
void extractBorders(cv::InputArray _binaryimg, cv::OutputArray _outbinaryimg, const cv::Size &_elsize=cv::Size(3,3))
{
    cv::Mat _tmpmat;
    cv::erode(_binaryimg,_tmpmat,cv::getStructuringElement(cv::MORPH_ELLIPSE,_elsize));
    _outbinaryimg.assign(_binaryimg.getMat() - _tmpmat);
}

#endif // OPENCVMORPH_H
