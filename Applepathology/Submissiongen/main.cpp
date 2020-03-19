#include <QDir>
#include <QFile>
#include <QStringList>
#include <QTextStream>

#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/dnn.h>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

const std::string options = "{samplesubmission     | | - file with sample submission}"
                            "{imagesdir            | | - directory where images are stored}"
                            "{networksdir          | | - directory where networks are stored}"
                            "{outputfile           | | - file where result should be saved}";

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdp(argc,argv,options);
    cmdp.about("This application was designed to generate submission for plat-pathology-2020 Kaggle competition");
    if(argc == 1) {
        cmdp.printMessage();
        return 0;
    }
    QStringList argnameslist;
    QStringList tmp = QString(options.c_str()).split('{',QString::SkipEmptyParts);
    for(int i = 0; i < tmp.size(); ++i) {
        argnameslist.push_back(tmp[i].section(' ',0,0));
        if(!cmdp.has(argnameslist[i].toStdString())) {
            qInfo("Argument '%s' is missed but required! Abort...", argnameslist[i].toUtf8().constData());
            return 1;
        }
    }

    QDir idir(cmdp.get<std::string>("imagesdir").c_str());
    if(!idir.exists()) {
        qInfo("Directory '%s' does not exist! Abort...", idir.absolutePath().toUtf8().constData());
        return 2;
    }

    QFile testfile(cmdp.get<std::string>("samplesubmission").c_str());
    if(!testfile.exists()) {
        qInfo("Samplesubmission '%s' does not exist! Abort...",testfile.fileName().toUtf8().constData());
        return 3;
    }
    testfile.open(QIODevice::ReadOnly);
    QStringList testfileslist;
    testfileslist.reserve(4096);
    QString submissionhead;
    while(!testfile.atEnd()) {
        QString line = testfile.readLine();
        if(line.contains("image_id"))
            submissionhead = qMove(line);
        else
            testfileslist.push_back(QString("%1.jpg").arg(line.section(',',0,0)));
    }
    qInfo("%d files has been declared to perform test", testfileslist.size());

    QDir ndir(cmdp.get<std::string>("networksdir").c_str());
    if(!ndir.exists()) {
        qInfo("Directory '%s' does not exist! Abort...", ndir.absolutePath().toUtf8().constData());
        return 4;
    }

    std::vector<dlib::softmax<dlib::anet_type::subnet_type>*> scabnetworksvector;
    std::vector<dlib::softmax<dlib::anet_type::subnet_type>*> rustnetworksvector;
    QStringList extensions;
    extensions << "*.dat";
    QStringList networksnames = ndir.entryList(extensions,QDir::Files | QDir::NoDotDot);
    for(const auto &filename: networksnames) {
        dlib::anet_type net;
        try {
            dlib::deserialize(ndir.absoluteFilePath(filename).toStdString()) >> net;
        } catch(std::exception& e) {
            qCritical("EXCEPTION IN LOADING MODEL DATA '%s' - %s", filename.toUtf8().constData(),e.what());
        }
        dlib::softmax<dlib::anet_type::subnet_type> *snet = new dlib::softmax<dlib::anet_type::subnet_type>();
        snet->subnet() = net.subnet();
        if(filename.contains("scab")) {
            scabnetworksvector.push_back(snet);
            qInfo(" - '%s' added to scab networks stack", filename.toUtf8().constData());
        } else if(filename.contains("rust")) {
            rustnetworksvector.push_back(snet);
            qInfo(" - '%s'  added to rust networks stack", filename.toUtf8().constData());
        }

    }

    QFile outputfile(cmdp.get<std::string>("outputfile").c_str());
    outputfile.open(QIODevice::WriteOnly | QIODevice::Truncate);
    QTextStream ts(&outputfile);
    ts.setRealNumberNotation(QTextStream::FixedNotation);
    ts.setRealNumberPrecision(7);
    ts << submissionhead;

    bool _isloaded;
    cv::Mat _tmpmat;
    dlib::rand rnd(777);
    const long crops = 8;

    for(const auto &imagefilename: testfileslist) {
        qInfo("Test for '%s'",imagefilename.toUtf8().constData());
        _tmpmat = loadIbgrmatWsize(idir.absoluteFilePath(imagefilename).toUtf8().constData(),
                                   IMG_WIDTH,IMG_HEIGHT,false,&_isloaded);
        if(!_isloaded) {
            qCritical("Can not open image '%s'",imagefilename.toUtf8().constData());
            assert(_isloaded);
        }

        dlib::matrix<dlib::rgb_pixel> _dlibtmpimg = cvmat2dlibmatrix<dlib::rgb_pixel>(_tmpmat);
        dlib::array<dlib::matrix<dlib::rgb_pixel>> _imgvariants;
        dlib::randomly_crop_image(_dlibtmpimg,_imgvariants,rnd,crops,0.7f,0.999f,0,0,true,true);

        std::vector<float> rustprobsvector, scabprobsvector;
        for(auto snet: scabnetworksvector) {
            dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat((*snet)(_imgvariants.begin(), _imgvariants.end())))/_imgvariants.size();
            scabprobsvector.push_back(p(0));
        }
        float scabprob = std::accumulate(scabprobsvector.begin(),scabprobsvector.end(),0.0f) / scabprobsvector.size();
        for(auto snet: rustnetworksvector) {
            dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat((*snet)(_imgvariants.begin(), _imgvariants.end())))/_imgvariants.size();
            rustprobsvector.push_back(p(1));
        }
        float rustprob = std::accumulate(rustprobsvector.begin(),rustprobsvector.end(),0.0f) / rustprobsvector.size();

        ts << imagefilename.section('.',0,0)    << ','
           << 1.0f - (rustprob + scabprob)/2.0f << ','
           << (rustprob + scabprob)/2.0f        << ','
           << rustprob                          << ','
           << scabprob                          << "\n";
        qInfo("   rust: %.3f\n   scab: %.3f", rustprob,scabprob);
        cv::imshow("probe",_tmpmat);
        cv::waitKey(0);
    }

    // Free allocated heap memory
    for(auto p: rustnetworksvector)
        delete p;
    for(auto p: scabnetworksvector)
        delete p;

    return 0;
}
