#include <QDir>
#include <QFile>
#include <QMap>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "dlibwhalesrecognizer.h"

using namespace std;

const cv::String keys = "{inputdir  i |       | input directory name (where dirty data is stored)}"
                        "{outputdir o |       | output directory name (where cleaned data should be stored)}"
                        "{model     m |       | model filename}"
                        "{dstthresh t | 0.595 | min distance for same ids}";

cv::Mat medianDescription(const vector<cv::Mat> &_vlbldscr)
{
    assert(_vlbldscr.size() > 0);

    if(_vlbldscr.size() == 1)
        return _vlbldscr[0].clone();

    cv::Mat _mediandscr = cv::Mat::zeros(_vlbldscr[0].rows,_vlbldscr[0].cols,CV_32FC1);
    for(size_t k = 0; k < _mediandscr.total(); ++k) {
        std::vector<float> _vtmp(_vlbldscr.size(),0.0f);
        for(size_t n = 0; n < _vtmp.size(); ++n)
            _vtmp[n] = _vlbldscr[n].at<const float>(k);
        std::nth_element(_vtmp.begin(),_vtmp.begin()+_vtmp.size()/2,_vtmp.end());
        _mediandscr.at<float>(k) = _vtmp[_vtmp.size()/2];
    }
    return _mediandscr;
}

size_t countValue(const vector<bool> _vin, bool _value)
{
    size_t _counter = 0;
    for(size_t i = 0; i < _vin.size(); ++i) {
        if(_vin[i] == _value)
            _counter++;
    }
    return _counter;
}

int main(int argc, char **argv)
{
    cv::CommandLineParser cmdparser(argc,argv,keys);
    cmdparser.about("This application allows you to semi-automatically filter train data");
    if(argc == 1) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("inputdir")) {
        qInfo("You have not provided input dir arg! Abort...");
        return 1;
    }
    if(!cmdparser.has("outputdir")) {
        qInfo("You have not provided output dir arg! Abort...");
        return 2;
    }
    if(!cmdparser.has("model")) {
        qInfo("You have not provided model filename! Abort...");
        return 3;
    }
    QDir qindir(cmdparser.get<string>("inputdir").c_str());
    if(!qindir.exists()) {
        qInfo("Input directory '%s' does not exist! Abort...", qindir.absolutePath().toUtf8().constData());
        return 4;
    }
    QDir qoutdir(cmdparser.get<string>("outputdir").c_str());
    if(qoutdir.exists()) {
        qInfo("Output directory '%s' already exist! Abort...", qoutdir.absolutePath().toUtf8().constData());
        return 5;
    }
    QFileInfo _modelfileinfo(cmdparser.get<string>("model").c_str());
    if(!_modelfileinfo.exists()) {
        qInfo("Model '%s' does not exist! Abort...",_modelfileinfo.filePath().toUtf8().constData());
        return 6;
    }
    double dstthresh = cmdparser.get<double>("dstthresh");

    // Load identification model
    cv::Ptr<cv::oirt::CNNImageRecognizer> recognizer = cv::oirt::createDlibWhalesRecognizer(cmdparser.get<cv::String>("model"));

    // Let's rock
    QStringList listofsubdirs = qindir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    QStringList filefilters;
    filefilters << "*.png" << "*.jpeg" << "*.bmp" << "*.jpg";
    qInfo("Stage I - 'in class' analysis");
    QMap<QString,QString> id2filenamemap;
    QMap<QString,cv::Mat> id2meddscrmap;
    for(int i = 0; i < listofsubdirs.size(); ++i) {
        qInfo("  lbl #%d (%s):", i, listofsubdirs.at(i).toUtf8().constData());
        QDir subdir(qindir.absolutePath().append(QString("/%1").arg(listofsubdirs.at(i))));
        QStringList listoffiles = subdir.entryList(filefilters, QDir::Files | QDir::NoDotAndDotDot);
        if(listoffiles.size() > 1) {
            vector<cv::Mat> _vlbldscr;
            _vlbldscr.reserve(listoffiles.size());
            vector<bool> _vpreservefile(listoffiles.size(),false);
            for(int j = 0; j < listoffiles.size(); ++j) {
                string _filename = subdir.absoluteFilePath(listoffiles.at(j)).toStdString();
                _vlbldscr.push_back(recognizer->getImageDescription(cv::imread(_filename,CV_LOAD_IMAGE_UNCHANGED)));
            }
            cv::Mat _mediandscr = medianDescription(_vlbldscr);
            // Compare with median description, preserve
            for(int j = 0; j < listoffiles.size(); ++j) {
                double _distance = cv::oirt::euclideanDistance(_mediandscr,_vlbldscr[j]);
                if(_distance < dstthresh) {
                    qInfo("    %.3f for %s ",_distance, listoffiles.at(j).toUtf8().constData());
                    _vpreservefile[j] = true;
                } else {
                    qInfo("    %.3f for %s - manual check is suggested",_distance, listoffiles.at(j).toUtf8().constData());
                    cv::imshow("Test picture",cv::imread(subdir.absoluteFilePath(listoffiles.at(j)).toStdString(),CV_LOAD_IMAGE_UNCHANGED));
                    cv::waitKey(30); // delay for the picture to be rendered properly
                    for(int k = 0; k < listoffiles.size(); ++k) {
                        if(cv::oirt::euclideanDistance(_mediandscr,_vlbldscr[k]) < dstthresh) {
                            cv::imshow("Reference picture",cv::imread(subdir.absoluteFilePath(listoffiles.at(k)).toStdString(),CV_LOAD_IMAGE_UNCHANGED));
                            break;
                        }
                    }
                    cv::waitKey(30); // delay for the picture to be rendered properly
                    cout << "Is this images belong to the same class? (yes/no or y/n): ";
                    string answer;
                    getline(cin,answer);
                    if((answer.compare("yes") == 0) || (answer.compare("y") == 0))
                        _vpreservefile[j] = true;
                    //cv::destroyAllWindows();
                }
            }
            // Also we should try to find identical images
            for(int k = 0; k < (listoffiles.size() - 1); ++k) {
                if(_vpreservefile[k]) {
                    for(int n = k+1; n < listoffiles.size(); ++n) {
                        if(_vpreservefile[n] && (cv::oirt::euclideanDistance(_vlbldscr[k],_vlbldscr[n]) < 0.01)) {
                            _vpreservefile[n] = false;
                        }
                    }
                }
            }
            // Now we can add data to maps
            int _validsamples = countValue(_vpreservefile,true);
            if(_validsamples > 1) {
                id2meddscrmap.insert(listofsubdirs.at(i),_mediandscr);
                //qInfo("    this files has been filtered for stage II:");
                for(int j = 0; j < listoffiles.size(); ++j) {
                    if(_vpreservefile[j]) {
                        id2filenamemap.insertMulti(listofsubdirs.at(i),subdir.absoluteFilePath(listoffiles.at(j)));
                        //qInfo("      %s", listoffiles.at(j).toUtf8().constData());
                    }
                }
            } else {
                qInfo("    insufficient number of valid samples (%d), label will be skipped", _validsamples);
            }
        } else {
            qInfo("    insufficient number of samples (%d), label will be skipped", listoffiles.size());
        }
    }

    qInfo("Stage II - 'between class' analysis");
    QStringList listofclasses = id2meddscrmap.keys();
    vector<bool> _vpreserveclass(listofclasses.size(),true);
    for(int i = 0; i < (listofclasses.size()-1); ++i) {
        if(_vpreserveclass[i]) {
            QStringList class1_fileslist = id2filenamemap.values(listofclasses.at(i));
            for(int j = i+1; j< listofclasses.size(); ++j) {
                if(_vpreserveclass[j]) {
                    double _distance = cv::oirt::euclideanDistance(id2meddscrmap.value(listofclasses.at(i)),id2meddscrmap.value(listofclasses.at(j)));
                    qInfo("    %.3f - #%d %s to #%d %s",_distance,i,listofclasses.at(i).toUtf8().constData(),j,listofclasses.at(j).toUtf8().constData());
                    if(_distance < dstthresh) {
                        if(_distance < 0.75*dstthresh) {
                            _vpreserveclass[j] = false;
                        } else {
                            QStringList class2_fileslist = id2filenamemap.values(listofclasses.at(j));
                            cv::imshow("Class 1", cv::imread(class1_fileslist.at(0).toStdString(),CV_LOAD_IMAGE_UNCHANGED));
                            cv::waitKey(30);
                            cv::imshow("Class 2", cv::imread(class2_fileslist.at(0).toStdString(),CV_LOAD_IMAGE_UNCHANGED));
                            cv::waitKey(30);
                            cout << "Is this images belong to the same class? (yes/no or y/n): ";
                            string answer;
                            getline(cin,answer);
                            if((answer.compare("yes") == 0) || (answer.compare("y") == 0))
                                _vpreserveclass[j] = false;
                            //cv::destroyAllWindows();
                        }
                    }
                }
            }
        }
    }

    // Create output directory
    qInfo("Wait untill filtered data will be copied to the output directory");
    qoutdir.mkpath(qoutdir.absolutePath());
    for(size_t i = 0; i < _vpreserveclass.size(); ++i) {
        if(_vpreserveclass[i]) {
            qoutdir.mkdir(listofclasses.at(i));
            QStringList fileslist = id2filenamemap.values(listofclasses.at(i));
            for(int j = 0; j < fileslist.size(); ++j) {
                QFile::copy(fileslist.at(j),qoutdir.absolutePath().append(QString("/%1/%2").arg(listofclasses.at(i),QFileInfo(fileslist.at(j)).fileName())));
            }
        }
        qInfo("%d) %s - %s", i, listofclasses.at(i).toUtf8().constData(), _vpreserveclass[i] ? "copied" : "dropped out");
    }
    qInfo("Done");
    return 0;
}
