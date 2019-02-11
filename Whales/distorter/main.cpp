#include <QDir>
#include <QUuid>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "opencvimgaugment.h"

cv::Mat distortperspective(const cv::Mat&_inmat, cv::RNG &_cvrng, double _maxportion=0.25, bool changesides=false, int _interp_method=CV_INTER_LINEAR, int _bordertype=cv::BORDER_DEFAULT)
{
    cv::Point2f pts1[]={
                        cv::Point2f(0,0),
                        cv::Point2f(_inmat.cols,0),
                        cv::Point2f(_inmat.cols,_inmat.rows),
                        cv::Point2f(0,_inmat.rows)
                       };
    float hshift = _cvrng.uniform(0.0,_maxportion)*_inmat.cols;
    float vshift = _cvrng.uniform(0.0,_maxportion)*_inmat.rows;
    if(changesides)
        vshift *= -1;
    cv::Point2f pts2[]={
                        cv::Point2f(hshift,- vshift),
                        cv::Point2f(_inmat.cols - hshift, vshift),
                        cv::Point2f(_inmat.cols - hshift, _inmat.rows - vshift),
                        cv::Point2f(hshift,_inmat.rows + vshift)
                       };
    cv::Mat _outmat;
    cv::warpPerspective(_inmat,_outmat,cv::getPerspectiveTransform(pts1,pts2),cv::Size(_inmat.cols,_inmat.rows),_interp_method,_bordertype,cv::mean(_inmat));
    _outmat = addNoise(_outmat,_cvrng,0,10);
    return _outmat;
}


const cv::String keys = "{dir i  | | directory to process}"
                        "{help h | | show help}";

int main(int argc, char **argv)
{
    cv::CommandLineParser cmdparser(argc,argv,keys);
    cmdparser.about("Run app to add distorted copies of images inside target directory");
    if(argc == 1 || cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(cmdparser.has("dir") == false) {
        qInfo("You have not provide directory to process! Abort...");
        return 1;
    }
    QDir indir(cmdparser.get<cv::String>("dir").c_str());
    if(indir.exists() == false) {
        qInfo("Target dir '%s' does not exist! Abort...", indir.absolutePath().toUtf8().constData());
        return 2;
    }

    QStringList filefilters;
    filefilters << "*.jpg" << "*.jpeg" << "*.png";

    QStringList list_of_subdirs = indir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    qInfo("Starting to make augmentation");
    cv::RNG cvrng;
    for(int i = 0; i < list_of_subdirs.size(); ++i) {
        QDir subdir(indir.absolutePath().append("/%1").arg(list_of_subdirs.at(i)));
        QStringList list_of_files = subdir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
        for(int j = 0; j < list_of_files.size(); ++j) {
            cv::Mat oimage = cv::imread(subdir.absoluteFilePath(list_of_files.at(j)).toStdString(),cv::IMREAD_UNCHANGED);
            cv::Mat timage = distortperspective(oimage,cvrng,0.25,false,CV_INTER_CUBIC,cv::BORDER_CONSTANT);
            cv::imwrite(subdir.absolutePath().append("/%1.jpg").arg(QUuid::createUuid().toString()).toStdString(),timage);
            timage = distortperspective(oimage,cvrng,0.25,true,CV_INTER_LINEAR,cv::BORDER_CONSTANT);
            cv::imwrite(subdir.absolutePath().append("/%1.jpg").arg(QUuid::createUuid().toString()).toStdString(),timage);
        }
        qInfo("  %d) %s - enrolled", i, list_of_subdirs.at(i).toUtf8().constData());
    }
    qInfo("Done");
    return 0;
}
