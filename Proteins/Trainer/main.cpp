#include <QDir>
#include <QFile>

#include <map>
#include <thread>

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

#include "customnetwork.h"

using namespace std;
using namespace dlib;

const cv::String keys =
   "{help h           |        | app help}"
   "{traindir t       |        | training directory location}"
   "{outputdir o      |        | output directory location}"
   "{number n         |   1    | number of classifiers to be trained}"
   "{swptrain         | 5000   | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
   "{swpvalid         | 1000   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
   "{minlrthresh      | 1.0e-5 | minimum learning rate, determines when training should be stopped}";

void fillLabelsMap();

int main(int argc, char** argv) try
{
    cv::CommandLineParser cmdparser(argc, argv, keys);
    cmdparser.about("This app has been developed for competition https://www.kaggle.com/c/human-protein-atlas-image-classification/data");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindirpath")) {
        qInfo("You have not provide path to training directory! Abort...");
        return 1;
    }
    if(!cmdparser.has("outputdirpath")) {
        qInfo("You have not provide path to output directory! Abort...");
        return 2;
    }
    if(cmdparser.get<int>("number") <= 0) {
        qInfo("Number of classifiers should be greater than zero! Abort...");
        return 3;
    }
    QDir traindir(cmdparser.get<string>("traindir").c_str());
    if(!traindir.exists()) {
        qInfo("Training directory does not exist! Abort...");
        return 4;
    }
    QFile trainfile(traindir.absoluteFilePath("train.csv"));
    if(!trainfile.exists()) {
        qInfo("Training dir does not contain train.csv! Abort...");
        return 5;
    }
    traindir.setPath(traindir.absolutePath().append("/train"));
    if(!traindir.exists()) {
        qInfo("Training dir does not contain /train subdir! Abort...");
        return 6;
    }
    // Ok, seems we have check everithing, now we can parse files
    fillLabelsMap();


    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {

    }

	return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

void fillLabelsMap()
{
    labels["0"] = {"y", "n"};
    labels["1"] = {"y", "n"};
    labels["2"] = {"y", "n"};
    labels["3"] = {"y", "n"};
    labels["4"] = {"y", "n"};
    labels["5"] = {"y", "n"};
    labels["6"] = {"y", "n"};
    labels["7"] = {"y", "n"};
    labels["8"] = {"y", "n"};
    labels["9"] = {"y", "n"};
    labels["10"] = {"y", "n"};
    labels["11"] = {"y", "n"};
    labels["12"] = {"y", "n"};
    labels["13"] = {"y", "n"};
    labels["14"] = {"y", "n"};
    labels["15"] = {"y", "n"};
    labels["16"] = {"y", "n"};
    labels["17"] = {"y", "n"};
    labels["18"] = {"y", "n"};
    labels["19"] = {"y", "n"};
    labels["20"] = {"y", "n"};
    labels["21"] = {"y", "n"};
    labels["22"] = {"y", "n"};
    labels["23"] = {"y", "n"};
    labels["24"] = {"y", "n"};
    labels["25"] = {"y", "n"};
    labels["26"] = {"y", "n"};
    labels["27"] = {"y", "n"};
}
