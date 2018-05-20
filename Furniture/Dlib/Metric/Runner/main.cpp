#include <opencv2/opencv.hpp>
#include <dlib/rand.h>

#include "furniturerecognizer.h"

int main(int argc, char** argv) try
{
    auto recognizer = cv::imgrec::createFurnitureRecognizer("/home/alex/Fastdata/Kaggle/Furniture/Metricbench/net.dat");
    recognizer->ImageRecognizer::load("/home/alex/Fastdata/Kaggle/Furniture/validlabels.yml");

    if(recognizer->empty()) {
        std::cout << "Can not load labels! Abort..." << std::endl;
        return 1;
    }

    std::string _path = "/home/alex/Fastdata/Kaggle/Furniture/Test/";
    cv::Mat _imgmat;
    dlib::rand rnd(time(0));

    std::ofstream ofs;
    ofs.open("/home/alex/Fastdata/Kaggle/Furniture/submission.csv");
    ofs << "id,predicted" << std::endl;

    std::string _filename, _labelname;
    for(int i = 1; i <= 12800; ++i) {
        _filename = std::to_string(i);
        std::cout << "\tfilename " << _filename << ".jpg";
        _imgmat = cv::imread(_path + _filename + ".jpg", cv::IMREAD_COLOR);
        if(!_imgmat.empty()) {
            int _lbl = recognizer->predict(_imgmat);
            _labelname = recognizer->getLabelInfo(_lbl);
            std::cout << " - label predicted: " << _labelname << std::endl;
        } else {
            _labelname = std::to_string(rnd.get_integer_in_range(1,128));
            std::cout << " - label guessed: " << _labelname << std::endl;
        }
        ofs << _filename << "," << _labelname << std::endl;
    }

    return 0;
}

catch(std::exception& e)
{
    std::cout << e.what() << std::endl;
}

