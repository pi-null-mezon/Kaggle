#ifndef OPENCVIMGALIGN_H
#define OPENCVIMGALIGN_H

#include <opencv2/opencv.hpp>

/// Applies complex image alignment: makes thresholding, finds center of mass, makes shifting to center of the mass, applies optional rotation to first PCA direction
cv::Mat alignPCAWResize(const cv::Mat &_inputmat, const cv::Mat &_mattotransform, cv::Size _targetsize=cv::Size(), double _thresh=0.0, int _interptype=cv::INTER_AREA, int _bordertype=cv::BORDER_REFLECT, bool _applyrotation=true);

/// Compose multiple channel image to one channel by making projection to first PCA direction
cv::Mat extractFirstPCAComponentImage(const std::vector<cv::Mat> &_vchannels, unsigned int _targetprojection=0);

#endif // OPENCVIMGALIGN_H
