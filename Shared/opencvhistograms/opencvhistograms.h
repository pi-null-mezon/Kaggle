#ifndef OPENCVHISTOGRAMS_H
#define OPENCVHISTOGRAMS_H

#include "opencv2/imgproc.hpp"

cv::Mat two_dimensional_histogram(cv::Mat bgr, int colorspace, const int *sizes, const int *channels, const float **ranges, bool normalize=true)
{
    cv::Mat mat;
    cv::cvtColor(bgr, mat, colorspace);
    cv::Mat hist;
    cv::calcHist(&mat, 1, channels, cv::Mat(), hist, 2, sizes, ranges);
    if(normalize)
        cv::normalize(hist, hist, 1, 0, cv::NORM_MINMAX);
    return hist;
}

cv::Mat HS_histogram(cv::Mat bgr, cv::Size size)
{
    int hbins = size.width, sbins = size.height;
    int hist_sizes[] = {hbins, sbins};
    int channels[] = {0, 1};
    float hue_ranges[] = { 0, 180 };
    float saturation_ranges[] = { 0, 256 };
    const float* ranges[] = { hue_ranges, saturation_ranges };
    return two_dimensional_histogram(bgr, cv::COLOR_BGR2HSV, hist_sizes, channels, ranges);
}

cv::Mat AB_histogram(cv::Mat bgr, cv::Size size)
{
    int hbins = size.width, sbins = size.height;
    int hist_sizes[] = {hbins, sbins};
    int channels[] = {1, 2};
    float a_ranges[] = { 0, 256 };
    float b_ranges[] = { 0, 256 };
    const float* ranges[] = { a_ranges, b_ranges };
    return two_dimensional_histogram(bgr, cv::COLOR_BGR2Lab, hist_sizes, channels, ranges);
}

cv::Mat CrCb_histogram(cv::Mat bgr, cv::Size size)
{
    int hbins = size.width, sbins = size.height;
    int hist_sizes[] = {hbins, sbins};
    int channels[] = {1, 2};
    float cr_ranges[] = { 0, 256 };
    float cb_ranges[] = { 0, 256 };
    const float* ranges[] = { cr_ranges, cb_ranges };
    return two_dimensional_histogram(bgr, cv::COLOR_BGR2YCrCb, hist_sizes, channels, ranges);
}


cv::Mat colors_histograms(cv::Mat bgr, cv::Size size=cv::Size(32,32))
{
    cv::Mat merged;
    std::vector<cv::Mat> planes;
    planes.push_back(HS_histogram(bgr,size));
    planes.push_back(AB_histogram(bgr,size));
    planes.push_back(CrCb_histogram(bgr,size));
    cv::merge(planes,merged);
    return merged;
}

#endif // OPENCVHISTOGRAMS_H
