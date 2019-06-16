#include <QDir>
#include <QFile>
#include <QStringList>

int main(int argc, char *argv[])
{
    if(argc == 1) {
        qInfo("This app was designed to transform data for Kaggle Kinship'2019 competittion");
        qInfo("Options:\n"
              "\n\t-i[str] - name of the input directory with original training data"
              "\n\t-f[str] - name of the 'train_relationships.csv' file"
              "\n\t-o[str] - name of the output directory");
    }
    QString inputdirname, outputdirname, pairsfile = "train_relationships.csv";
    while(--argc && (**(++argv) == '-')) {
        char opt = *(++argv[0]);
        switch(opt) {
            case 'i':
                inputdirname = ++argv[0];
                break;
            case 'f':
                pairsfile = ++argv[0];
                break;
            case 'o':
                outputdirname = ++argv[0];
                break;
        }
    }
    if(inputdirname.isEmpty()) {
        qInfo("Input directory name is not specified! Abort...");
        return 1;
    }
    if(outputdirname.isEmpty()) {
        qInfo("Output directory name is not specified! Abort...");
        return 2;
    }
    QDir idir(inputdirname);
    if(!idir.exists()) {
        qInfo("Input directory '%s' does not exist! Abort...", idir.absolutePath().toUtf8().constData());
        return 3;
    }
    QDir odir(outputdirname);
    if(odir.exists()) {
        qInfo("Output directory '%s' already exist! Abort...", odir.absolutePath().toUtf8().constData());
        return 4;
    }
    odir.mkpath(outputdirname);
    QFile pfile(pairsfile);
    if(!pfile.exists()) {
        qInfo("File with pairs kinship '%s' can not be found! Abort...", pairsfile.toUtf8().constData());
        return 5;
    }
    qInfo("Input directory: %s\n",inputdirname.toUtf8().constData());
    qInfo("Output directory: %s\n", outputdirname.toUtf8().constData());
    bool _isopened = pfile.open(QIODevice::ReadOnly);
    qInfo("Pairs kinship file %s", _isopened ? "has been opened" : "can not be opened!");
    QStringList filefilters;
    filefilters << "*.jpg" << "*.png";
    unsigned long counter = 0;
    while(!pfile.atEnd()) {
        counter++;
        QString line = QString(pfile.readLine());
        QString left = line.section(',',0,0);
        QString right = line.section(',',1,1).trimmed();
        qInfo(" Files for %s - %s are under processing",left.toUtf8().constData(),right.toUtf8().constData());
        QDir leftdir(idir.absolutePath().append("/%1").arg(left));
        QDir rightdir(idir.absolutePath().append("/%1").arg(right));
        if(leftdir.exists() && rightdir.exists()) {
            QStringList leftfiles = leftdir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
            QStringList rightfiles = rightdir.entryList(filefilters,QDir::Files | QDir::NoDotAndDotDot);
            if(leftfiles.size() > 0 && rightfiles.size() > 0) {
                odir.mkdir(QString::number(counter));
                foreach (const auto &filename, leftfiles)
                    QFile::copy(leftdir.absoluteFilePath(filename),odir.absolutePath().append("/%1/%2").arg(QString::number(counter),filename));
                foreach (const auto &filename, rightfiles)
                    QFile::copy(rightdir.absoluteFilePath(filename),odir.absolutePath().append("/%1/%2").arg(QString::number(counter),filename));
            }
        }
    }
    qInfo("All files has been copied");
    return 0;
}
