#include <iostream>

#include <QFile>
#include <QDebug>
#include <QDir>

#include <opencv2/opencv.hpp>

#include "opencvimgalign.h"

struct RadarImage {
    RadarImage(int _size=75) {
        size = _size;
        band1.resize(_size);
        band2.resize(_size);
        inc_angle = QString();
        is_iceberg = false;
    }
    cv::String          idcvstr;
    int                 size;
    std::vector<float>  band1;
    std::vector<float>  band2;
    cv::Mat             mat;
    QString             inc_angle;
    int                 is_iceberg;
};

cv::Mat logdbtransform(const cv::Mat &_mat);

void                parseHugeJSON(const QByteArray &_ba, std::vector<RadarImage> &_radarimgvector, bool _visualize=true);
std::vector<float>  extractBandImage(const QString &_source, const QString &_bandname);
QString             extractID(const QString &_source, const QString &_key="id");
QString extractAngle(const QString &_source, const QString &_key="inc_angle");
int                 extractIsIceberg(const QString &_source, const QString &_key="is_iceberg");

int main(int argc, char *argv[])
{
    if(argc != 3) {
        qWarning() << "Please pass input file as first cmd arg and output directory as second";
        return 1;
    }

    QDir _dir;
    _dir.setPath(argv[2]);
    if(_dir.exists() == false) {
        if(_dir.mkpath(argv[2]) == false) {
            qWarning() << "Can not create output directory " << argv[2] << "! Abort...";
            return 2;
        }
    }
    _dir.cd(argv[2]);
    _dir.mkdir("Iceberg");
    _dir.mkdir("Ship");

    QFile _file(argv[1]);
    if(_file.open(QFile::ReadOnly)) {
        qInfo() << "File processing has been started. Please wait...";
        /* Standard Qt JSON file parser can not handle with such 'large' document :(
        QByteArray _ba = _file.readAll();
        QJsonParseError _jsonparsererror;
        QJsonDocument jsondoc = QJsonDocument::fromJson(_ba, &_jsonparsererror);
        qDebug() << _jsonparsererror.errorString();
        qInfo() << jsondoc.isArray();
        */

        // So, we will handle it by our bare hands
        QByteArray _ba = _file.readAll();
        std::vector<RadarImage> _vradarimg;
        parseHugeJSON(_ba, _vradarimg, false);
        qInfo() << "Images has been extracted " << _vradarimg.size() << ". Start saving images as files, please wait...";
        QString _absdirname_for_icebergs = _dir.absolutePath().append("/Iceberg/");
        QString _absdirname_for_ships = _dir.absolutePath().append("/Ship/");
        for(size_t i = 0; i < _vradarimg.size(); ++i) {
            if(_vradarimg[i].is_iceberg) {
                cv::imwrite((_absdirname_for_icebergs + QString("%1.png").arg(_vradarimg[i].idcvstr.c_str())).toLocal8Bit().constData(), _vradarimg[i].mat);
            } else {
                cv::imwrite((_absdirname_for_ships + QString("%1.png").arg(_vradarimg[i].idcvstr.c_str())).toLocal8Bit().constData(), _vradarimg[i].mat);
            }
        }
    } else {
        qWarning() << "Can not open file " << argv[1] << "! Abort...";
        return 3;
    }

    return 0;
}


void parseHugeJSON(const QByteArray &_ba, std::vector<RadarImage> &_radarimgvector, bool _visualize)
{
    int pos1 = 0;
    int pos2 = 0;
    for(int i = 0; i < _ba.size(); ++i) {
        if(_ba[i] == '{') {
            pos1 = i;
        }
        if(_ba[i] == '}') {
            static unsigned int imgnum = 0;
            pos2 = i;
            QByteArray _tmpba;
            _tmpba.resize(pos2-pos1);
            std::memcpy((void*)(&_tmpba.data()[0]),(void*)(&_ba.data()[pos1]),pos2-pos1);
            QString _str(_tmpba);
            RadarImage _tmpradimg;
            _tmpradimg.idcvstr = extractID(_str).toUtf8().constData();
            _tmpradimg.band1 = std::move(extractBandImage(_str,"band_1"));
            _tmpradimg.band2 = std::move(extractBandImage(_str,"band_2"));
            _tmpradimg.inc_angle = extractAngle(_str);
            _tmpradimg.is_iceberg = extractIsIceberg(_str);


            qInfo() << "Image #" << imgnum++;
            qInfo() << "id:" << _tmpradimg.idcvstr.c_str();
            qInfo() << "band_1 area:" << _tmpradimg.band1.size();
            qInfo() << "band_2 area:" << _tmpradimg.band2.size();
            qInfo() << "inc_angle:" << _tmpradimg.inc_angle;
            qInfo() << "is_iceberg:" << _tmpradimg.is_iceberg;

            double _min,_max;
            cv::Mat _imgmat1(_tmpradimg.size,_tmpradimg.size,CV_32FC1,&_tmpradimg.band1[0]);
            //_imgmat1 = logdbtransform(_imgmat1);
            cv::minMaxIdx(_imgmat1,&_min,&_max);
            qInfo() << "min_1: " <<  _min << "\tmax_1: " << _max;
            cv::Mat _vchannelmean, _vchannelstdev;
            cv::meanStdDev(_imgmat1,_vchannelmean,_vchannelstdev);
            qInfo() << "mean_1:" << _vchannelmean.at<const double>(0) << "\tstdev_1:" << _vchannelstdev.at<const double>(0);
            cv::Mat _nmat1 = (_imgmat1 -_vchannelmean.at<const double>(0)) / _vchannelstdev.at<const double>(0);

            cv::Mat _imgmat2(_tmpradimg.size,_tmpradimg.size,CV_32FC1,&_tmpradimg.band2[0]);
            //_imgmat2 = logdbtransform(_imgmat2);
            cv::minMaxIdx(_imgmat2,&_min,&_max);
            qInfo() << "min_2: " <<  _min << "\tmax_2: " << _max;
            cv::meanStdDev(_imgmat2,_vchannelmean,_vchannelstdev);
            qInfo() << "mean_2:" << _vchannelmean.at<const double>(0) << "\tstdev_2:" << _vchannelstdev.at<const double>(0);
            cv::Mat _nmat2 = (_imgmat2 - _vchannelmean.at<const double>(0)) / _vchannelstdev.at<const double>(0);

            cv::Mat _imgmat3;
            std::vector<cv::Mat> _vc;
            _vc.push_back(_imgmat1);
            _vc.push_back(_imgmat2);
            _imgmat3 = projectToPCA(_vc);
            cv::minMaxIdx(_imgmat3,&_min,&_max);
            qInfo() << "min_3: " <<  _min << "\tmax_3: " << _max;
            cv::meanStdDev(_imgmat3,_vchannelmean,_vchannelstdev);
            qInfo() << "mean_3:" << _vchannelmean.at<const double>(0) << "\tstdev_3:" << _vchannelstdev.at<const double>(0);
            cv::Mat _nmat3 = (_imgmat3 - _vchannelmean.at<const double>(0)) / _vchannelstdev.at<const double>(0);

            _nmat1.convertTo(_nmat1, CV_8U, 17.5, 104.298);
            _nmat2.convertTo(_nmat2, CV_8U, 17.5, 117.001);
            _nmat3.convertTo(_nmat3, CV_8U, 17.5, 122.782);

            std::vector<cv::Mat> _channels;
            _channels.push_back(_nmat1);           
            _channels.push_back(_nmat2);
            _channels.push_back(_nmat3);

            cv::Mat _coloredmat;
            cv::merge(_channels, _coloredmat);
            //_coloredmat = alignByPCAWithCrop(_coloredmat,cv::Size(70,70),165,cv::INTER_LANCZOS4,cv::BORDER_REFLECT,false);

            /*cv::threshold(_nmat3,_nmat3,0.0,255.0,CV_THRESH_OTSU | CV_THRESH_TOZERO);
            cv::Scalar _ss = cv::sum(_nmat3);
            qInfo() << "sum: " << _ss[0];*/

            _tmpradimg.mat = _visualize ? _coloredmat.clone() : _coloredmat;
            if(_tmpradimg.inc_angle.length() < 8)
                _radarimgvector.push_back(std::move(_tmpradimg));

            // Sample of how to save 16 bit depth one channel png
            /*_vc.clear();
            _vc.push_back(_imgmat1);
            _vc.push_back(_imgmat2);
            _imgmat3 = projectToPCA(_vc);
            cv::normalize(_imgmat3,_imgmat3,0.0,65535.0,cv::NORM_MINMAX);
            _imgmat3.convertTo(_nmat3,CV_16U);
            _tmpradimg.mat = _visualize ? _nmat3.clone() : _nmat3;
            _radarimgvector.push_back(std::move(_tmpradimg));*/

            if(_visualize) {

                    cv::namedWindow("Band_1", CV_WINDOW_NORMAL);
                    cv::imshow("Band_1", _nmat1);

                    cv::namedWindow("Band_2", CV_WINDOW_NORMAL);
                    cv::imshow("Band_2", _nmat2);

                    cv::namedWindow("MixedBand", CV_WINDOW_NORMAL);
                    cv::imshow("MixedBand", _nmat3);

                cv::putText(_coloredmat, _tmpradimg.is_iceberg ? "iceberg" : "ship", cv::Point(6,11), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1, CV_AA);
                cv::putText(_coloredmat, _tmpradimg.is_iceberg ? "iceberg" : "ship", cv::Point(5,10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1, CV_AA);
                cv::namedWindow("Colored", CV_WINDOW_NORMAL);
                cv::imshow("Colored", _coloredmat);
                cv::waitKey(0);
            }
            qInfo() << "-------------------";                       
        }
    }
}


std::vector<float> extractBandImage(const QString &_source, const QString &_bandname)
{
    QStringList _vallist = _source.section(_bandname,1).section('[',1).section(']',0,0).split(',');
    std::vector<float> _radarimgvector(_vallist.size(), 0.0f);
    for(int i = 0; i < _vallist.size(); ++i) {
        _radarimgvector[i] = _vallist.at(i).toFloat();
    }
    return _radarimgvector;
}

QString extractID(const QString &_source, const QString &_key)
{
    return _source.section(_key,1).section(':',1).section('"',1).section('"',0,0);
}

QString extractAngle(const QString &_source, const QString &_key)
{
    return _source.section(_key,1).section(':',1).section(',',0,0);
}

int extractIsIceberg(const QString &_source, const QString &_key)
{
    return _source.section(_key,1).section(':',1).contains("1") ? 1 : 0;
}

cv::Mat logdbtransform(const cv::Mat &_imat) {
    if(_imat.type() != CV_32F)
        CV_Error(cv::Error::BadDepth, "Input image shoud have float type!");
    if(_imat.channels() != 1)
        CV_Error(cv::Error::BadDepth, "Input image shoud hhave single channel!");
    if(_imat.isContinuous() != true)
        CV_Error(cv::Error::BadDepth, "Input image shoud be continuous!");

    cv::Mat _omat(_imat.cols,_imat.rows,CV_32F);
    const float *_in = _imat.ptr<const float>(0);
    float *_out = _omat.ptr<float>(0);
    for(int i = 0; i < _imat.rows*_imat.cols; ++i) {
        if(_in[i] > 0.0f)
            _out[i] = _in[i];
        else
            _out[i] = -std::log(1.0f - _in[i]);
    }
    return _omat;
}
