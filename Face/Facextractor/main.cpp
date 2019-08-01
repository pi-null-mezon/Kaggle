#include <QStringList>
#include <QUuid>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "facetracker.h"

const cv::String _options = "{help h            |     | this help                           }"
                            "{inputdir i        |     | input directory with images         }"
                            "{outputdir o       |     | output directory with images        }"
                            "{faceshapemodel m  |     | Dlib's face shape model file        }"
                            "{facedetcascade c  |     | opencv's face detector (if not specified Dlib's HOG detector will be used)}"
                            "{targetwidth       | 150 | target image width                  }"
                            "{targetheight      | 150 | target image height                 }"
                            "{hportion          | 1.2 | target horizontal face portion      }"
                            "{vportion          | 1.2 | target vertical face portion        }"
                            "{hshift            | 0   | target horizontal face shift        }"
                            "{vshift            | 0   | target vertical face shift          }"
                            "{visualize v       |false| enable/disable visualization option }";

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif

    cv::CommandLineParser _cmdparser(argc,argv,_options);
    if(_cmdparser.has("help")) {
        _cmdparser.printMessage();
        return 0;
    }

    if(!_cmdparser.has("inputdir")) {
        qWarning("You have not pointed an input directory! Abort...");
        return 1;
    }
    QDir _indir(_cmdparser.get<cv::String>("inputdir").c_str());
    QStringList _filters;
    _filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    QStringList _fileslist = _indir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);
    qInfo("There is %d pictures has been found in the input directory", _fileslist.size());

    if(!_cmdparser.has("outputdir")) {
        qWarning("You have not pointed output directory! Abort...");
        return 2;
    }
    QDir _outdir(_cmdparser.get<cv::String>("outputdir").c_str());
    if(_outdir.exists() == false) {
        _outdir.mkpath(_cmdparser.get<cv::String>("outputdir").c_str());
        _outdir.cd(_cmdparser.get<cv::String>("outputdir").c_str());
    }

    if(!_cmdparser.has("faceshapemodel")) {
        qWarning("You have not pointed dlib's face shape model file! Abort...");
        return 3;
    }
    dlib::shape_predictor _faceshapepredictor;
    try {
        dlib::deserialize(_cmdparser.get<cv::String>("faceshapemodel").c_str()) >> _faceshapepredictor;
    } catch(const std::exception& e) {
        qWarning("%s", e.what());
        return 4;
    }

    cv::CascadeClassifier _facedetector;
    FaceTracker::PrimaryFaceDetector detectortype = FaceTracker::ViolaJones;
    if(!_cmdparser.has("facedetcascade")) {
        qWarning("You have not pointed opencv's face detector cascade file, so Dlib's HOG detector will be used...");
        detectortype = FaceTracker::HOG;
    } else {
        _facedetector.load(_cmdparser.get<cv::String>("facedetcascade"));
        if(_facedetector.empty()) {
            qWarning("Empty face detector cascade classifier! Abort...");
            return 5;
        }
    }

    FaceTracker _facetracker(1,FaceTracker::FaceShape);
    _facetracker.setFaceShapePredictor(&_faceshapepredictor);
    if(detectortype == FaceTracker::ViolaJones)
        _facetracker.setFaceClassifier(&_facedetector);
    _facetracker.setPrimaryFaceDetectorType(detectortype);
    _facetracker.setFaceRectPortions(_cmdparser.get<float>("hportion"),_cmdparser.get<float>("vportion"));
    _facetracker.setFaceRectShifts(_cmdparser.get<float>("hshift"),_cmdparser.get<float>("vshift"));

    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    int _facenotfound = 0;
    const bool _visualize = _cmdparser.get<bool>("visualize");
    for(int i = 0; i < _fileslist.size(); ++i) {
        cv::Mat _imgmat = cv::imread(_indir.absoluteFilePath(_fileslist.at(i)).toLocal8Bit().constData());
        cv::Mat _facemat = _facetracker.getResizedFaceImage(_imgmat,_targetsize);
        if(!_facemat.empty()) {
            qInfo("%d) %s - face found", i, _fileslist.at(i).toUtf8().constData());
            if(_visualize) {
                cv::imshow("Probe",_facemat);
                cv::waitKey(1);
            }
            cv::imwrite(QString("%1/%2.jpg").arg(_outdir.absolutePath(),QUuid::createUuid().toString()).toUtf8().constData(),_facemat);
        } else {
            qInfo("%d) %s - could not find face!", i, _fileslist.at(i).toUtf8().constData());
            _facenotfound++;
        }
    }
    qInfo("Work has been acomplished successfully. Faces found: %d / %d", _fileslist.size() - _facenotfound, _fileslist.size());

    return 0;
}
