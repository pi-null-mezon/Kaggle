#include <QDir>
#include <QFile>
#include <iostream>

using namespace std;

float random_portion()
{
    return static_cast<float>(qrand()) / RAND_MAX;
}

int main(int argc, char **argv)
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif
    char *inputdirname=nullptr, *outputdirname=nullptr;
    float trainingportion = 0.75f;
    while((--argc > 0) && ((*++argv)[0] == '-'))
        switch(*++argv[0]) {
        case 'i':
                inputdirname = ++argv[0];
            break;
        case 'o':
                outputdirname = ++argv[0];
            break;
        case 'p':
                trainingportion = QString(++argv[0]).toFloat();
            break;
        case 'h':
            qInfo("This application should be used for dataset division on training and validation parts");
            qInfo(" -i[str]   - input directory name (all data will be preserved)");
            qInfo(" -o[str]   - where validation and training data should be saved");
            qInfo(" -p[real] - portion of training data (default 0.75)");
            return 0;
        }
    if(inputdirname == nullptr) {
        qWarning("Empty input directory argument! Abort...");
        return 1;
    }
    if(outputdirname == nullptr) {
        qWarning("Empty output directory argument! Abort...");
        return 2;
    }
    QDir inputdir(inputdirname);
    if(!inputdir.exists()) {
        qWarning("Input directory does not exists! Abort...");
        return 1;
    }
    QDir outputdir(outputdirname);
    if(!inputdir.exists()) {
        qWarning("Output directory does not exists! Abort...");
        return 1;
    }

    auto subdirs = inputdir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for(int i = 0; i < subdirs.size(); ++i) {
        outputdir.mkpath(QString("Validation/%1").arg(subdirs.at(i)));
        outputdir.mkpath(QString("Training/%1").arg(subdirs.at(i)));
        QDir subdir(inputdir.absolutePath().append("/%1").arg(subdirs.at(i)));
        auto files = subdir.entryList(QDir::Files | QDir::NoDotAndDotDot);
        for(int j = 0; j < files.size(); ++j) {
            QString _targetfilename = outputdir.absolutePath();
            if(random_portion() < trainingportion) {
                _targetfilename.append("/Training");
            } else {
                _targetfilename.append("/Validation");
            }
            _targetfilename.append(QString("/%1/%2").arg(subdirs.at(i),files.at(j)));
            QFile::copy(subdir.absoluteFilePath(files.at(j)), _targetfilename);
            qInfo("%s -> %s", subdir.absoluteFilePath(files.at(j)).toUtf8().constData(), _targetfilename.toUtf8().constData());
        }
    }
    qInfo("Done");
    return 0;
}
