#include <QDir>
#include <QTextStream>

#include "dlibimgaugment.h"

#include "customnetwork.h"

int main(int argc, char *argv[])
{
    QString _indirname, _outfilename, _modelfilename;
    while((--argc > 0) && ((*++argv)[0] == '-'))
        switch(*++argv[0]) {
            case 'h':
                qInfo(" -h      - help");
                qInfo(" -i[str] - input directory name");
                qInfo(" -o[str] - output file name");
                qInfo(" -m[str] - model file name");
            break;

            case 'i':
                _indirname = ++argv[0];
            break;

            case 'o':
                _outfilename = ++argv[0];
            break;

            case 'm':
                _modelfilename = ++argv[0];
            break;
        }

    qInfo("Input dir name: %s", _indirname.toUtf8().constData());
    qInfo("Output file name: %s", _outfilename.toUtf8().constData());
    if(_indirname.isEmpty()) {
        qInfo("Empty input dir name! Abort...");
        return 1;
    }
    if(_outfilename.isEmpty()) {
        qInfo("Empty output file name! Abort...");
        return 2;
    }
    if(_modelfilename.isEmpty()) {
        qInfo("Empty model file name! Abort...");
        return 3;
    }
    QFileInfo _modelfileinfo(_modelfilename);
    if(_modelfileinfo.exists() == false) {
        qInfo("Model file does not exist! Abort...");
        return 4;
    }
    QDir indir(_indirname);
    if(!indir.exists()) {
        qInfo("Input directory does not exist! Abort...");
        return 5;
    }
    QFile _outfile(_outfilename);
    if(_outfile.open(QFile::WriteOnly) == false) {
        qInfo("Could not open output file for write! Abort...");
        return 6;
    }
    QTextStream _ots(&_outfile);
    _ots << "Id,Predicted\n";

    anet_type _net;
    try {
        dlib::deserialize(_modelfilename.toStdString()) >> _net;
    } catch(std::exception& e) {
        cout << e.what() << endl;
    }

    QStringList _filesfilter;
    _filesfilter << "*.png" << "*.tiff";
    QStringList _fileslist = indir.entryList(_filesfilter, QDir::Files | QDir::NoDotAndDotDot);
    qInfo("%d files has been found. Wait untill processing will be accomplished...", _fileslist.size());
    for(int i = 0; i < _fileslist.size(); ++i) {
        if(_fileslist.at(i).contains("_green")) {
            bool _is_file_loaded = false;
            std::map<std::string,dlib::loss_multimulticlass_log_::classifier_output> _pmap = _net(load_grayscale_image_with_normalization(indir.absoluteFilePath(_fileslist.at(i)).toStdString(),IMG_SIZE,IMG_SIZE,false, &_is_file_loaded));
            assert(_is_file_loaded);
            _ots << _fileslist.at(i).section("_green",0,0) << ',';
            std::map<std::string,dlib::loss_multimulticlass_log_::classifier_output>::const_iterator _it;
            bool _first_prediction = true;
            for(_it = _pmap.begin(); _it != _pmap.end(); ++_it) {
                std::string _predictedlbl = _it->second;
                if(_predictedlbl.compare("y") == 0) {
                    if(_first_prediction) {
                        _first_prediction = false;
                        _ots << _it->first.c_str();
                    } else {
                        _ots << ' ' << _it->first.c_str();
                    }
                }
            }
            _ots << '\n';
        }
    }
    qInfo("All predictions ready");
    return 0;
}
