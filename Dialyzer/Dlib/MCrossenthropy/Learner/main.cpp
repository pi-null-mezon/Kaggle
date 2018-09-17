// The contents of this file are in the public domain

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include "dlibimgaugment.h"

using namespace std;
using namespace dlib;

#define CLASSES 128
#define IMGSIZE 400

// ----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = res<64,res<64,res_down<64,SUBNET>>>;
template <typename SUBNET> using level4 = res<32,res<32,res_down<32,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares_down<64,SUBNET>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares_down<32,SUBNET>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            relu<affine<con<16,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>;

// training network type
using net_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            relu<bn_con<con<16,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>;

// ----------------------------------------------------------------------------------------

template <typename anynet_type, size_t layernum>
void set_learning_rate_multiplier(anynet_type &_anynet, double _value)
{
    layer<layernum>(_anynet).layer_details().set_learning_rate_multiplier(_value);
    layer<layernum>(_anynet).layer_details().set_bias_learning_rate_multiplier(_value);
}

class MyVisitor {
public:
    template <typename any_net>
    void operator()(size_t _idx, any_net &_layer) {
        _layer.layer_details().set_learning_rate_multiplier(0);
        _layer.layer_details().set_bias_learning_rate_multiplier(0);
    }
};

// -----------------------------------------------------------------------------------------

struct image_info
{
    string filename;
    string label;
    long numeric_label;
};

void get_files_listing(const std::string& images_folder, std::vector<image_info>& info)
{
    image_info temp;
    temp.numeric_label = 0;
    // We will loop over all the label types in the dataset, each is contained in a subfolder.
    auto subdirs = directory(images_folder).get_dirs();
    // But first, sort the sub directories so the numeric labels will be assigned in sorted order.
    std::sort(subdirs.begin(), subdirs.end());
    /*cout << "Note that labels will be presented in following order: " << endl;
    for(size_t i = 0; i < subdirs.size(); ++i)
        cout << "label " << i << " - class name " <<   subdirs[i].name() << endl;*/

    cout << "Subdirs found: " << subdirs.size() << endl;
    for (auto subdir : subdirs)  {
        // Now get all the images in this label type
        temp.label = subdir.name();
        for (auto image_file : subdir.get_files()) {
            temp.filename = image_file;
            info.push_back(temp);
        }
        ++temp.numeric_label;
    }
}

// ----------------------------------------------------------------------------------------
const cv::String keys =
       "{help h           |        | print this message}"
       "{traindirpath   t |        | training directory location}"
       "{validdirpath   v |        | validation directory location}"
       "{outputdirpath  o |        | output directory location}"
       "{number n         |   1    | number of classifiers to be trained}"
       "{lossthresh       | 0.20   | testset loss threshold for network saving}"
       "{swptrain         | 7500   | determines after how many steps without progress (training loss) decay should be applied to learning rate}"
       "{swptest          | 1000   | determines after how many steps without progress (test loss) decay should be applied to learning rate}"
       "{minlrthresh      | 1.0e-4 | minimum learning rate, determines when trining should be stopped}";
// -----------------------------------------------------------------------------------------
int main(int argc, char** argv) try
{
    cv::CommandLineParser cmdparser(argc, argv, keys);
    cmdparser.about("Application name dnn_classifier_learner");
    if(cmdparser.has("help")) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("traindirpath")) {
        cout << "You have not provide path to training directory!";
        return 2;
    }
    if(!cmdparser.has("validdirpath")) {
        cout << "You have not provide path to validation directory!";
        return 2;
    }
    if(!cmdparser.has("outputdirpath")) {
        cout << "You have not provide path to output directory!";
        return 2;
    }
    if(cmdparser.get<int>("number") <= 0) {
        cout << "Number of classifiers should be greater than 0!";
        return 2;
    }

    for(int n = 0; n < cmdparser.get<int>("number"); ++n) {
        cout << "\nNET #" << n << "\n------------------------------------------------------" << endl;

        std::vector<image_info> trainingset, validationset;
        get_files_listing(cmdparser.get<std::string>("traindirpath"), trainingset);
        get_files_listing(cmdparser.get<std::string>("validdirpath"), validationset);
        cout << "Training data split (train / test): " << trainingset.size() << " / " << validationset.size() << endl;
        const auto number_of_classes = trainingset.back().numeric_label + 1;
        cout << "Number of classes in training set: " << number_of_classes << endl;
        if(trainingset.size() == 0 || validationset.size() == 0 || number_of_classes != CLASSES)    {
            cout << "Didn't find dataset or dataset size split is wrong!" << endl;           
            return 1;
        }

        net_type net;
        /*anet_type resnet34(); // as in original imagenet set
        deserialize("/home/alex/Programming/3rdParties/dlib/data/resnet34_1000_imagenet_classifier.dnn") >> resnet34;
        layer<3>(net) = layer<3>(resnet34); // copy all except first 0,1,2 layers*/
        set_all_bn_running_stats_window_sizes(net, number_of_classes);

        /*cout << "The net has " << net.num_layers << " layers in it." << endl;
        set_learning_rate_multiplier<anet_type,143>(net,0);
        set_learning_rate_multiplier<anet_type,138>(net,0);
        set_learning_rate_multiplier<anet_type,135>(net,0);
        set_learning_rate_multiplier<anet_type,130>(net,0);
        set_learning_rate_multiplier<anet_type,127>(net,0);
        set_learning_rate_multiplier<anet_type,122>(net,0);
        set_learning_rate_multiplier<anet_type,119>(net,0);
        set_learning_rate_multiplier<anet_type,114>(net,0);
        set_learning_rate_multiplier<anet_type,111>(net,0);
        set_learning_rate_multiplier<anet_type,103>(net,0);
        set_learning_rate_multiplier<anet_type,100>(net,0);
        set_learning_rate_multiplier<anet_type,95>(net,0);
        set_learning_rate_multiplier<anet_type,92>(net,0);
        set_learning_rate_multiplier<anet_type,87>(net,0);
        set_learning_rate_multiplier<anet_type,84>(net,0);
        set_learning_rate_multiplier<anet_type,79>(net,0);
        set_learning_rate_multiplier<anet_type,76>(net,0);
        set_learning_rate_multiplier<anet_type,68>(net,0);
        set_learning_rate_multiplier<anet_type,65>(net,0);
        set_learning_rate_multiplier<anet_type,60>(net,0);
        set_learning_rate_multiplier<anet_type,57>(net,0);
        set_learning_rate_multiplier<anet_type,52>(net,0);
        set_learning_rate_multiplier<anet_type,49>(net,0);
        set_learning_rate_multiplier<anet_type,44>(net,0);
        set_learning_rate_multiplier<anet_type,41>(net,0);
        set_learning_rate_multiplier<anet_type,36>(net,0);
        set_learning_rate_multiplier<anet_type,33>(net,0);
        set_learning_rate_multiplier<anet_type,28>(net,0);
        set_learning_rate_multiplier<anet_type,25>(net,0);
        set_learning_rate_multiplier<anet_type,17>(net,0);
        set_learning_rate_multiplier<anet_type,14>(net,0);
        set_learning_rate_multiplier<anet_type,9>(net,0);
        set_learning_rate_multiplier<anet_type,6>(net,0);
        set_learning_rate_multiplier<anet_type,1>(net,2);
        cout << net << endl;
        return 0;*/

        dnn_trainer<net_type> trainer(net,sgd(0.0001, 0.9));
        trainer.be_verbose();
        trainer.set_learning_rate(0.1);
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdirpath") + "/trainer_" + std::to_string(n) + "_state.dat", std::chrono::minutes(15));
        trainer.set_learning_rate(0.01);
        // This threshold is probably excessively large.  You could likely get good results
        // with a smaller value but if you aren't in a hurry this value will surely work well.
        trainer.set_iterations_without_progress_threshold(cmdparser.get<int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<int>("swptest"));

        std::vector<matrix<rgb_pixel>> samples, validationsamples;
        std::vector<unsigned long> labels, validationlabels;

        // Start a bunch of threads that read images from disk and pull out random crops.  It's
        // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
        // thread for this kind of data preparation helps us do that.  Each thread puts the
        // crops into the data queue.
        dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> data(1024);
        auto f = [&data, &trainingset, number_of_classes](time_t seed)
        {
            dlib::rand rnd(time(0)+seed);
            matrix<rgb_pixel> img;
            dlib::array<matrix<rgb_pixel>> crops;
            std::pair<image_info, matrix<rgb_pixel>> temp;           
            while(data.is_enabled())
            {
                temp.first = trainingset[rnd.get_random_32bit_number()%trainingset.size()];
                dlib::load_image(img,temp.first.filename);
                /*dlib::disturb_colors(img,rnd);
                if(rnd.get_random_float() > 0.5f)
                    img = fliplr(img);
                randomly_jitter_image(img,crops,rnd.get_integer(INT_MAX),1,IMGSIZE,IMGSIZE,1.2,0.05,15.0);*/
                randomly_crop_image(img,crops,rnd,1,0.95,0.99,IMGSIZE,IMGSIZE);
                /*if(rnd.get_random_float() > 0.1f)
                    randomly_cutout_rect(crops[0],crops,rnd,1,0.5,0.5);*/
                temp.second = std::move(crops[0]);
                data.enqueue(temp);
            }
        };
        std::thread data_loader1([f](){ f(1); });
        std::thread data_loader2([f](){ f(2); });
#ifdef DLIB_USE_CUDA        
        std::thread data_loader3([f](){ f(3); });
        std::thread data_loader4([f](){ f(4); });
        std::thread data_loader5([f](){ f(5); });
        std::thread data_loader6([f](){ f(6); });
        std::thread data_loader7([f](){ f(7); });
        std::thread data_loader8([f](){ f(7); });
#endif

        dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> validationdata(256);
        auto vf = [&validationdata, &validationset, number_of_classes](time_t seed)
        {
            dlib::rand rnd(time(0)+seed);
            std::pair<image_info, matrix<rgb_pixel>> temp;
            while(validationdata.is_enabled())
            {
                temp.first = validationset[rnd.get_random_32bit_number()%validationset.size()];
                temp.second = std::move(load_rgb_image_with_fixed_size(temp.first.filename,IMGSIZE,IMGSIZE,false));
                validationdata.enqueue(temp);
            }
        };
        std::thread data_loader9([vf](){ vf(1); });
#ifdef DLIB_USE_CUDA
        std::thread data_loader10([vf](){ vf(2); });
#endif

        // The main training loop.  Keep making mini-batches and giving them to the trainer.
        // We will run until the learning rate has dropped to 1e-4 or number of steps exceeds 1e5
        const double _min_learning_rate_thresh = cmdparser.get<double>("minlrthresh");
#ifdef DLIB_USE_CUDA
        const size_t _training_minibatch_size = 79;
        const size_t _test_minibatch_size = 79;
#else
        const size_t _training_minibatch_size = 128;
        const size_t _test_minibatch_size = 32;
#endif
        while(trainer.get_learning_rate() > _min_learning_rate_thresh)
        {
            std::pair<image_info, matrix<rgb_pixel>> img;
            samples.clear();
            labels.clear();
            // make mini-batch           
            while(samples.size() < _training_minibatch_size) {
                data.dequeue(img);
                samples.push_back(std::move(img.second));
                labels.push_back(img.first.numeric_label);
            }
            trainer.train_one_step(samples, labels);

            if(trainer.get_train_one_step_calls() % 10 == 0) { // Now we can perform validation test
                validationsamples.clear();
                validationlabels.clear();               
                while(validationsamples.size() < _test_minibatch_size) {
                    validationdata.dequeue(img);
                    validationsamples.push_back(std::move(img.second));
                    validationlabels.push_back(img.first.numeric_label);
                }
                trainer.test_one_step(validationsamples,validationlabels);
            }
        }
        if(trainer.get_learning_rate() < _min_learning_rate_thresh)
            cout << "Learning rate has decreased below the threshold. Learning has been accomplished" << endl;

        // Training done, tell threads to stop and make sure to wait for them to finish before
        // moving on.
        data.disable();
        validationdata.disable();
        data_loader1.join();
        data_loader2.join();
        data_loader9.join();
#ifdef DLIB_USE_CUDA        
        data_loader3.join();
        data_loader4.join();
        data_loader5.join();
        data_loader6.join();
        data_loader7.join();
        data_loader8.join();
        data_loader10.join();
#endif

        // Also wait for threaded processing to stop in the trainer.
        trainer.get_net();
        net.clean();
        cout << "Network #" << n << " (trainset loss: " << trainer.get_average_loss() << "; test loss: " << trainer.get_average_test_loss() << ")" << endl;
        if(trainer.get_average_test_loss() < cmdparser.get<double>("lossthresh")) {           
            // Now test the network on the validation dataset.  First, make a testing
            // network with softmax as the final layer.  We don't have to do this if we just wanted
            // to test the "top1 accuracy" since the normal network outputs the class prediction.
            // But this snet object will make getting the top5 predictions easy as it directly
            // outputs the probability of each class as its final output.
            anet_type anet = net; // batch norm -> affine
            softmax<anet_type::subnet_type> snet;
            snet.subnet() = anet.subnet();
            cout << "Testing network on train dataset..." << endl;
            int num_right_top1 = 0;
            int num_wrong_top1 = 0;
            dlib::rand rnd(time(0));
            // loop over all the imagenet validation images
            double logloss = 0.0;
            for (auto l : validationset) {
                matrix<rgb_pixel> img;
                load_image(img,l.filename);
                // Grab N random crops from the image.  We will run all of them through the
                // network and average the results.
                dlib::array<matrix<rgb_pixel>> images;
                const size_t num_crops = 7;
                randomly_crop_image(img,images,rnd,num_crops,0.85,0.99,IMGSIZE,IMGSIZE,true,true);
                matrix<float,1,CLASSES> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
                // p(i) is the probability that the image contains object of class i.
                // update log loss
                logloss += -std::log(p(l.numeric_label));
                // check top 1 accuracy
                if(index_of_max(p) == l.numeric_label) {
                    ++num_right_top1;
                } else {
                    ++num_wrong_top1;
                    if(cmdparser.get<int>("number") == 1) {
                        cv::Mat _imgmat = dlibmatrix2cvmat(img);
                        cv::putText(_imgmat, string("true:") + l.label, cv::Point(6,11), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,0), 1, CV_AA);
                        cv::putText(_imgmat, string("true:") + l.label, cv::Point(5,10), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1, CV_AA);
                        cv::namedWindow("Wrong", CV_WINDOW_NORMAL);
                        cv::imshow("Wrong", _imgmat);
                        cv::waitKey(1);
                        cout << "True label: " << l.numeric_label << "; "
                             << "predicted: " << index_of_max(p) << endl;
                    }
                }
            }
            logloss /= validationset.size();
            cout << "Average testset log loss:  " << logloss << endl;
            cout << "Test accuracy: " << 1.0 - (double)num_wrong_top1/(num_right_top1+num_wrong_top1) << endl;
            cout << "Wrong / Total: " << num_wrong_top1 << "/" << (num_right_top1+num_wrong_top1) << endl;
            serialize(cmdparser.get<std::string>("outputdirpath") + "/net_" + std::to_string(n) + "_(" + std::to_string(trainer.get_average_loss()) + " - " + std::to_string(trainer.get_average_test_loss()) + " - " + std::to_string(logloss) + ").dat") << net;
        } else {
            cout << "Training session is not well enough, this network will not be saved..." << endl;
        }
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

