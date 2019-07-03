#include <QDir>
#include <QFileInfo>
#include <QTextStream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "dlibopencvconverter.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"


const cv::String options = "{m model         |       | - name to cnn model file}"
                           "{f facedscrmodel |       | - path to dlib_face_recognition_resnet_model_v1.dat}"
                           "{t testdir       |       | - name of the test directory}"
                           "{p pairs         |       | - path to the sample_submission.csv file}"
                           "{o output        |       | - name of the output file}"
                           "{v               | false | - data visualization}"
                           "{h help          |       | - flag to show help}";

int main(int argc, char *argv[])
{
    cv::CommandLineParser _cmdp(argc,argv,options);
    _cmdp.about("This app was designed to participate in Kaggle Kinship competition, summer of 2019");
    if(argc == 1 || _cmdp.has("help")) {
        _cmdp.printMessage();
        return 0;
    }

    if(!_cmdp.has("model")) {
        qInfo("You have not provided model for prediction! Abort...");
        return 1;
    }
    if(!QFileInfo(_cmdp.get<cv::String>("model").c_str()).exists()) {
        qInfo("Model file '%s' does not exist! Abort...",_cmdp.get<cv::String>("model").c_str());
        return 2;
    }
    if(!_cmdp.has("facedscrmodel")) {
        qInfo("You have not provided model for face descriptor! Abort...");
        return 11;
    }
    qInfo("Trying to load face descriptor model, please wait...");
    dlib::dlib_face_dscr_type fdscrnet;
    try{
        dlib::deserialize(_cmdp.get<string>("facedscrmodel"))>> fdscrnet;
    } catch(std::exception &e) {
        qInfo("EXCEPTION WHILE LOADING FACE DESCRIPTOR MODEL");
        qInfo("%s",e.what());
    }
    qInfo("Success");
    if(!_cmdp.has("testdir")) {
        qInfo("You have not provided testdir for prediction! Abort...");
        return 3;
    }
    QDir _testdir(_cmdp.get<cv::String>("testdir").c_str());
    if(!_testdir.exists()) {
        qInfo("Test directory you have provided '%s' does not exist! Abort...", _cmdp.get<cv::String>("testdir").c_str());
        return 4;
    }
    if(!_cmdp.has("pairs")) {
        qInfo("You have not provided file with sample submission pairs! Abort...");
        return 5;
    }
    if(!QFileInfo(_cmdp.get<cv::String>("pairs").c_str()).exists()) {
        qInfo("File with sample submission pairs '%s' does not exist! Abort...",_cmdp.get<cv::String>("pairs").c_str());
        return 6;
    }
    if(!_cmdp.has("output")) {
        qInfo("You have not provided output filename! Abort...");
        return 7;
    }
    QFile _outputfile(_cmdp.get<cv::String>("output").c_str());
    if(!_outputfile.open(QIODevice::WriteOnly)) {
        qInfo("Can not open file '%s' to write! Abort...", _outputfile.fileName().toUtf8().constData());
        return 8;
    }
    QTextStream _ots(&_outputfile);
    _ots.setRealNumberNotation(QTextStream::FixedNotation);
    _ots.setRealNumberPrecision(6);

    bool _visualize = _cmdp.get<bool>("v");

    QStringList _filesfilters;
    _filesfilters << "*.jpg" << "*.png";
    QStringList _imgfiles = _testdir.entryList(_filesfilters,QDir::Files | QDir::NoDotAndDotDot);

    qInfo("Trying to load model, please wait...");
    dlib::anet_type net;
    try {
        dlib::deserialize(_cmdp.get<cv::String>("model").c_str()) >> net;
    } catch(std::exception& e) {
        cout << "EXCEPTION IN LOADING MODEL DATA" << endl;
        cout << e.what() << endl;
    }
    dlib::softmax<dlib::anet_type::subnet_type> snet;
    snet.subnet() = net.subnet();
    qInfo("Success");

    QFile _pairsfile(_cmdp.get<cv::String>("pairs").c_str());
    _pairsfile.open(QIODevice::ReadOnly);
    unsigned long _lines = 0;
    bool _isloaded;   
    while(!_pairsfile.atEnd()) {
        QString _line(_pairsfile.readLine());
        if(_lines == 0) {
            _ots << _line;
        } else {

            cv::Mat _leftmat = loadIbgrmatWsize(_testdir.absoluteFilePath(_line.section('-',0,0)).toStdString(),
                                                  IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            if(!_isloaded) {
                qInfo("  WARNING: file '%s' can not be loaded!",_line.section('-',0,0).toUtf8().constData());
                continue;
            }

            cv::Mat _rightmat = loadIbgrmatWsize(_testdir.absoluteFilePath(_line.section('-',1,1).section(',',0,0)).toStdString(),
                                                   IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
            if(!_isloaded) {
                qInfo("  WARNING: file '%s' can not be loaded!",_line.section('-',1,1).section(',',0,0).toUtf8().constData());
                continue;
            }

            dlib::matrix<float,0,1> _leftdscr = fdscrnet(cvmat2dlibmatrix<dlib::rgb_pixel>(_leftmat));
            dlib::matrix<float,0,1> _rightdscr = fdscrnet(cvmat2dlibmatrix<dlib::rgb_pixel>(_rightmat));
            dlib::matrix<float,0,1> _features;
            _features.set_size(dlib::num_rows(_leftdscr));
            for(int i = 0; i < dlib::num_rows(_leftdscr); ++i)
                _features(i) = (_leftdscr(i)-_rightdscr(i))*(_leftdscr(i)-_rightdscr(i));

            dlib::matrix<float,1,2> _prob = dlib::mat(snet(_features));
            float _kinshipsprob = _prob(1);

            qInfo("  %lu) kinship prob estimation: %.4f",_lines,_kinshipsprob);
            if(_visualize) {
                cv::imshow("left",loadIbgrmatWsize(_testdir.absoluteFilePath(_line.section('-',0,0)).toStdString(),IMG_WIDTH,IMG_HEIGHT,false));
                cv::imshow("right",loadIbgrmatWsize(_testdir.absoluteFilePath(_line.section('-',1,1).section(',',0,0)).toStdString(),IMG_WIDTH,IMG_HEIGHT,false));
                if(_kinshipsprob > 0.5f)
                    cv::waitKey(0);
                else
                    cv::waitKey(1);
            }
            _ots << _line.section(',',0,0) << ',' << _kinshipsprob << "\n";
        }
        _lines++;
    }
    qInfo("Done");
    return 0;
}
