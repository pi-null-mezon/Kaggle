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
#define IMGSIZE 190

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

template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128,res<128,res<128,res_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

// training network type
using net_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<32,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc<CLASSES,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,5,5,2,2,
                            input_rgb_image_sized<IMGSIZE>
                            >>>>>>>>>>>;
//-----------------------------------------------------------------------------------------

struct image_info
{
    string filename;
    string label;
    long numeric_label;
};

void get_training_files_listing(const std::string& images_folder, std::vector<image_info>& traininfo, std::vector<image_info>& validinfo, float _validation_part)
{
    image_info temp;
    temp.numeric_label = 0;
    // We will loop over all the label types in the dataset, each is contained in a subfolder.
    auto subdirs = directory(images_folder).get_dirs();
    // But first, sort the sub directories so the numeric labels will be assigned in sorted order.
    std::sort(subdirs.begin(), subdirs.end());
    dlib::rand rnd(time(0));
    for (auto subdir : subdirs)  {
        // Now get all the images in this label type
        temp.label = subdir.name();
        for (auto image_file : subdir.get_files()) {
            temp.filename = image_file;
            if(rnd.get_random_float() < _validation_part)
                validinfo.push_back(temp);
            else
                traininfo.push_back(temp);
        }
        ++temp.numeric_label;
    }
}


dlib::matrix<dlib::rgb_pixel> load_rgb_image_with_fixed_size(string _filename, int _trows, int _tcols)
{   
    cv::Mat _originalimgmat = cv::imread(_filename, CV_LOAD_IMAGE_COLOR);
    if(_originalimgmat.cols > _tcols || _originalimgmat.rows > _trows)
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,CV_INTER_AREA);
    else if(_originalimgmat.cols < _tcols || _originalimgmat.rows < _trows)
        cv::resize(_originalimgmat,_originalimgmat,cv::Size(_tcols,_trows),0,0,CV_INTER_LINEAR);
    return cvmat2dlibmatrix<dlib::rgb_pixel>(_originalimgmat);
}

// ----------------------------------------------------------------------------------------
const cv::String keys =
       "{help h           |        | print this message   }"
       "{traindirpath   t |        | training directory location   }"
       "{outputdirpath  o |        | output directory location   }"
       "{number n         |   1    | number of classifiers to be trained   }"
       "{split s          | 0.05   | test portion of train data   }"
       "{lossthresh       | 0.20   | testset loss threshold for network saving }"
       "{swptrain         | 10000  | determines after how many steps without progress (training loss) decay should be applied to learning rate  }"
       "{swptest          | 2000   | determines after how many steps without progress (test loss) decay should be applied to learning rate  }"
       "{minlrthresh      | 1.0e-3 | minimum learning rate, determines when trining should be stopped  }";
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

        std::vector<image_info> trainingset, validationtset;
        get_training_files_listing(cmdparser.get<std::string>("traindirpath"), trainingset, validationtset, cmdparser.get<float>("split"));        
        cout << "Training data split (train / test): " << trainingset.size() << " / " << validationtset.size() << endl;
        const auto number_of_classes = trainingset.back().numeric_label+1;
        if(trainingset.size() == 0 || validationtset.size() == 0 || number_of_classes != CLASSES)    {
            cout << "Didn't find the Kaggle iceberg dataset or dataset size split is wrong!" << endl;
            return 1;
        }

        net_type net;
        dnn_trainer<net_type> trainer(net,sgd(0.0001, 0.9));
        trainer.be_verbose();
        trainer.set_learning_rate(0.1);
        trainer.set_synchronization_file(cmdparser.get<std::string>("outputdirpath") + "/trainer_" + std::to_string(n) + "_state.dat", std::chrono::minutes(15));
        // This threshold is probably excessively large.  You could likely get good results
        // with a smaller value but if you aren't in a hurry this value will surely work well.
        trainer.set_iterations_without_progress_threshold(cmdparser.get<int>("swptrain"));
        trainer.set_test_iterations_without_progress_threshold(cmdparser.get<int>("swptest"));
        // Since the progress threshold is so large might as well set the batch normalization
        // stats window to something big too.
        set_all_bn_running_stats_window_sizes(net, number_of_classes);

        std::vector<matrix<rgb_pixel>> samples, validationsamples;
        std::vector<unsigned long> labels, validationlabels;

        // Start a bunch of threads that read images from disk and pull out random crops.  It's
        // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
        // thread for this kind of data preparation helps us do that.  Each thread puts the
        // crops into the data queue.
        dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> data(2048);
        auto f = [&data, &trainingset, number_of_classes](time_t seed)
        {
            dlib::rand rnd(time(0)+seed);
            matrix<rgb_pixel> img;
            dlib::array<matrix<rgb_pixel>> crops;
            std::pair<image_info, matrix<rgb_pixel>> temp;
            while(data.is_enabled())
            {
                long _classes = 0;
                std::vector<bool> _vsoc_selected(number_of_classes, false);
                while(_classes < number_of_classes) {
                    temp.first = trainingset[rnd.get_random_32bit_number()%trainingset.size()];
                    if(_vsoc_selected[temp.first.numeric_label] == false) {
                        _vsoc_selected[temp.first.numeric_label] = true;
                        _classes++;
                        img = std::move(load_rgb_image_with_fixed_size(temp.first.filename,IMGSIZE,IMGSIZE));
                        dlib::disturb_colors(img,rnd);
                        /*size_t num_crops = 1;
                        if(rnd.get_random_float() > 0.5f) {
                            img = fliplr(img);
                        }
                        if(rnd.get_random_float() > 0.1f) {
                            randomly_jitter_image(img,crops,seed,num_crops,0,0,1.1,0.05,11.0);
                            img = std::move(crops[0]);
                            if(rnd.get_random_float() > 0.2f) {
                                randomly_cutout_rect(img,crops,rnd,num_crops);
                                img = std::move(crops[0]);
                            }
                        }*/
                        temp.second = std::move(img);
                        data.enqueue(temp);
                    }
                }
            }
        };
        std::thread data_loader1([f](){ f(1); });
#ifdef DLIB_USE_CUDA
        std::thread data_loader2([f](){ f(2); });
        std::thread data_loader3([f](){ f(3); });
        std::thread data_loader4([f](){ f(4); });
        std::thread data_loader5([f](){ f(5); });
        std::thread data_loader6([f](){ f(6); });
        std::thread data_loader7([f](){ f(7); });
        std::thread data_loader8([f](){ f(8); });
        std::thread data_loader9([f](){ f(9); });
        std::thread data_loader10([f](){ f(10); });
        std::thread data_loader11([f](){ f(11); });
        std::thread data_loader12([f](){ f(12); });
        std::thread data_loader13([f](){ f(13); });
#endif

        dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> validationdata(512);
        auto vf = [&validationdata, &validationtset, number_of_classes](time_t seed)
        {
            dlib::rand rnd(time(0)+seed);
            matrix<rgb_pixel> img;
            std::pair<image_info, matrix<rgb_pixel>> temp;
            while(validationdata.is_enabled())
            {
                long _classes = 0;
                std::vector<bool> _vsoc_selected(number_of_classes, false);
                while(_classes < number_of_classes) {
                    temp.first = validationtset[rnd.get_random_32bit_number()%validationtset.size()];
                    if(_vsoc_selected[temp.first.numeric_label] == false) {
                        _vsoc_selected[temp.first.numeric_label] = true;
                        _classes++;
                        img = std::move(load_rgb_image_with_fixed_size(temp.first.filename,IMGSIZE,IMGSIZE));
                        /*dlib::array<matrix<rgb_pixel>> crops;
                        size_t num_crops = 1;
                        if(rnd.get_random_float() > 0.1f) {// take in mind that this is validation images preprocessing
                            randomly_jitter_image(img,crops,seed,num_crops,0,0,1.1,0.03,9.0);
                            img = std::move(crops[0]);
                        }*/
                        temp.second = std::move(img);
                        validationdata.enqueue(temp);
                    }
                }
            }
        };
        std::thread data_loader14([vf](){ vf(1); });
#ifdef DLIB_USE_CUDA
        std::thread data_loader15([vf](){ vf(2); });
        std::thread data_loader16([vf](){ vf(3); });
        std::thread data_loader17([vf](){ vf(4); });
#endif

        // The main training loop.  Keep making mini-batches and giving them to the trainer.
        // We will run until the learning rate has dropped to 1e-4 or number of steps exceeds 1e5
        const double _min_learning_rate_thresh = cmdparser.get<double>("minlrthresh");
#ifdef DLIB_USE_CUDA
        const size_t _training_minibatch_size = 128;
        const size_t _test_minibatch_size = 32;
#else
        const size_t _training_minibatch_size = 128;
        const size_t _test_minibatch_size = 32;
#endif
        while(trainer.get_learning_rate() > _min_learning_rate_thresh)
        {
            samples.clear();
            labels.clear();
            // make mini-batch
            std::pair<image_info, matrix<rgb_pixel>> img;
            while(samples.size() < _training_minibatch_size) {
                data.dequeue(img);
                samples.push_back(std::move(img.second));
                labels.push_back(img.first.numeric_label);
            }
            trainer.train_one_step(samples, labels);

            if(trainer.get_train_one_step_calls() % 40 == 0) { // Now we can perform validation test
                validationsamples.clear();
                validationlabels.clear();
                std::pair<image_info, matrix<rgb_pixel>> validationimg;
                while(validationsamples.size() < _test_minibatch_size) {
                    validationdata.dequeue(validationimg);
                    validationsamples.push_back(std::move(validationimg.second));
                    validationlabels.push_back(validationimg.first.numeric_label);
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
        data_loader14.join();
#ifdef DLIB_USE_CUDA
        data_loader2.join();
        data_loader3.join();
        data_loader4.join();
        data_loader5.join();
        data_loader6.join();
        data_loader7.join();
        data_loader8.join();
        data_loader9.join();
        data_loader10.join();
        data_loader11.join();
        data_loader12.join();
        data_loader13.join();

        data_loader15.join();
        data_loader16.join();
        data_loader17.join();
#endif

        // Also wait for threaded processing to stop in the trainer.
        trainer.get_net();
        net.clean();
        cout << "Network #" << n << " (trainset loss: " << trainer.get_average_loss() << "; test loss: " << trainer.get_average_test_loss() << ")" << endl;
        if(trainer.get_average_test_loss() < cmdparser.get<double>("lossthresh")) {
            //serialize(cmdparser.get<std::string>("outputdirpath") + "/net_" + std::to_string(n) + "_(" + std::to_string(trainer.get_average_loss()) + " - " + std::to_string(trainer.get_average_test_loss()) + ").dat") << net;
            // Now test the network on the validation dataset.  First, make a testing
            // network with softmax as the final layer.  We don't have to do this if we just wanted
            // to test the "top1 accuracy" since the normal network outputs the class prediction.
            // But this snet object will make getting the top5 predictions easy as it directly
            // outputs the probability of each class as its final output.
            softmax<anet_type::subnet_type> snet; snet.subnet() = net.subnet();
            cout << "Testing network on train dataset..." << endl;
            int num_right_top1 = 0;
            int num_wrong_top1 = 0;
            dlib::rand rnd(time(0));
            // loop over all the imagenet validation images
            double logloss = 0.0;
            for (auto l : validationtset)
            {
                dlib::array<matrix<rgb_pixel>> images;
                matrix<rgb_pixel> img = std::move(load_rgb_image_with_fixed_size(l.filename,IMGSIZE,IMGSIZE));;
                // Grab N random crops from the image.  We will run all of them through the
                // network and average the results.
                const size_t num_crops = 5;
                randomly_crop_image(img,images,rnd,num_crops,IMGSIZE*0.9f,IMGSIZE*0.9f);
                matrix<float,1,CLASSES> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;
                // p(i) == the probability the image contains object of class i.
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
                        cv::waitKey(33);
                        cout << "True label: " << l.numeric_label << "; "
                             << "predicted: " << index_of_max(p) << endl;
                    }
                }
            }
            logloss /= validationtset.size();
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

