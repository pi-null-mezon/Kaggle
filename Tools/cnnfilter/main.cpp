#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <dlib/dnn.h>

#include "opencvimgaugment.h"
#include "dlibimgaugment.h"
#include "dlibopencvconverter.h"

#include "../../BKK/Occlusion/customnetwork.h"

const std::string options = "{inputdir i  |       | - directory with files to be checked}"
                            "{outputdir o |       | - directory where filtered files should be copied}"
                            "{resources r |       | - directory where CNN's *.dat files are stored}"
                            "{batchsize b | 256   | - batch size to process}"
                            "{label l     |   0   | - label of the class that should be preserved}"
                            "{help h      |       | - help}";


int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("App should be used to filter pictures by means of CNN-enrollment");
    if(argc == 1 || cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }

    if(!cmdparser.has("inputdir")) {
        qInfo("Empty input directory name! Abort...");
        return 1;
    }
    QDir qindir(cmdparser.get<std::string>("inputdir").c_str());
    if(!qindir.exists()) {
        qInfo("Input directory '%s' does not exist! Abort...", qindir.absolutePath().toUtf8().constData());
        return 2;
    }

    if(!cmdparser.has("outputdir")) {
        qInfo("Empty output directory name! Abort...");
        return 3;
    }

    if(!cmdparser.has("resources")) {
        qInfo("Empty resources directory name! Abort...");
        return 4;
    }
    QDir qresdir(cmdparser.get<std::string>("resources").c_str());
    if(!qresdir.exists()) {
        qInfo("Resources directory '%s' does not exist! Abort...", qresdir.absolutePath().toUtf8().constData());
        return 5;
    }
    QStringList resourceslist = qresdir.entryList(QStringList() << "*.dat", QDir::Files | QDir::NoDotAndDotDot);
    if(resourceslist.size() == 0) {
        qInfo("No resources *.dat files found in '%s'! Abort...", qresdir.absolutePath().toUtf8().constData());
        return 6;
    }   

    qInfo("Files reading, please wait...");
    QStringList fileslist = qindir.entryList(QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp", QDir::Files | QDir::NoDotAndDotDot);
    QStringList validfileslist;
    validfileslist.reserve(fileslist.size());
    bool isloaded = false;
    for(const QString &_filename : fileslist) {
        const QString _absfilename = qindir.absoluteFilePath(_filename).toUtf8().constData();
        dlib::matrix<dlib::rgb_pixel> picture = dlib::load_rgb_image_with_fixed_size(_absfilename.toStdString(),IMG_WIDTH,IMG_HEIGHT,false,&isloaded);
        if(!isloaded)
            qInfo("  file '%s' can not be loaded :(", _filename.toUtf8().constData());
        else
            validfileslist.push_back(_absfilename);
    }
    if(validfileslist.size() > 0) {
        qInfo("Resources initialization, please wait...");
        std::vector<dlib::softmax<dlib::anet_type::subnet_type>> snets(resourceslist.size());
        int i = 0;
        for(auto &filename: resourceslist) {
            qInfo("  deserialization of '%s'", filename.toUtf8().constData());
            try {
                dlib::anet_type _tmpnet;
                dlib::deserialize(qresdir.absoluteFilePath(filename).toStdString()) >> _tmpnet;
                snets[i++].subnet() = _tmpnet.subnet();
            } catch(const std::exception& e) {
                qInfo(" !!!! exception while model loading: %s", e.what());
                return 7;
            }
        }

        qInfo("Files checking, please wait...");
        int label = cmdparser.get<int>("label");
        std::vector<float> labelproblist(validfileslist.size(),0);

        const int batchsize = cmdparser.get<int>("batchsize");
        int batches = validfileslist.size() / batchsize;
        for(int b = 0; b < batches + 1; ++b) {
            std::vector<dlib::matrix<dlib::rgb_pixel>> pictures;
            pictures.reserve(batchsize);
            if(b < batches) {
                for(int i = 0; i < batchsize; ++i) {
                    pictures.push_back(dlib::load_rgb_image_with_fixed_size(validfileslist.at(b * batchsize + i).toStdString(),IMG_WIDTH,IMG_HEIGHT,false));
                }
            } else {
                for(int i = 0; i < (validfileslist.size() - b * batchsize); ++i) {
                    pictures.push_back(dlib::load_rgb_image_with_fixed_size(validfileslist.at(b * batchsize + i).toStdString(),IMG_WIDTH,IMG_HEIGHT,false));
                }
            }


            for(size_t i = 0; i < snets.size(); ++i) {
                auto predictions = dlib::mat(snets[i](pictures.begin(),pictures.end()));
                //qInfo("prediction size: %ld x %ld", dlib::num_rows(predictions), dlib::num_columns(predictions));
                for(long j = 0; j < dlib::num_rows(predictions); ++j) {
                    labelproblist[b * batchsize + j] += predictions(j,label);
                }
            }
        }

        QDir qoutdir(cmdparser.get<std::string>("outputdir").c_str());
        if(!qoutdir.exists())
            qoutdir.mkpath(qoutdir.absolutePath());

        for(size_t i = 0; i < labelproblist.size(); ++i) {
            float prob = labelproblist[i] / snets.size();
            const QString filename = validfileslist.at(i).section('/',-1,-1);
            if(prob >= 0.5f)
                QFile::copy(validfileslist.at(i),qoutdir.absolutePath().append("/%1").arg(filename));
            else
                qInfo("  file '%s' will not be preserved prob(%d) ~ %.4f", filename.toUtf8().constData(), label, prob);
        }
    } else {
        qInfo("No valid files to check found");
    }
    qInfo("Done");
    return 0;
}
