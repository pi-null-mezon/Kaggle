#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

const std::string options = "{inputdir i |       | - directory with files to be checked}"
                            "{delete d   | false | - delete files that can not be opened}";


int main(int argc, char *argv[])
{
    cv::CommandLineParser cmdparser(argc,argv,options);
    cmdparser.about("Application should be used to check files if they can be correctly opened by opencv's imread method");
    if(argc == 1) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("inputdir")) {
        qInfo("Empty input directory name! Abort...");
        return 1;
    }
    const bool _delete = cmdparser.get<bool>("delete");

    QDir qdir(cmdparser.get<std::string>("inputdir").c_str());
    if(!qdir.exists()) {
        qInfo("Input directory '%s' does not exist! Abort...", qdir.absolutePath().toUtf8().constData());
        return 2;
    }
    QStringList files = qdir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    for(const QString &_filename : files) {
        const QString _absolutefilename = qdir.absoluteFilePath(_filename).toUtf8().constData();
        qInfo("Check '%s'", _absolutefilename.toUtf8().constData());
        cv::Mat _tmpmat = cv::imread(_absolutefilename.toUtf8().constData(),cv::IMREAD_UNCHANGED);
        if(_tmpmat.empty()) {
            qInfo(" - can not be opened");
            if(_delete) {
                if(QFile::remove(_absolutefilename))
                    qInfo(" - removed");
                else
                    qInfo(" - can not be removed");
            }
        }
    }
    return 0;
}
