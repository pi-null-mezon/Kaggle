#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

using namespace std;
using namespace dlib;
//-----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<max_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;
// ----------------------------------------------------------------------------------------
#define FNUM 64
template <typename SUBNET> using level2 = res_down<FNUM*5,SUBNET>;
template <typename SUBNET> using level3 = res<FNUM*4,res_down<FNUM*4,SUBNET>>;
template <typename SUBNET> using level4 = res_down<8*FNUM,SUBNET>;
template <typename SUBNET> using level5 = res<4*FNUM,res_down<4*FNUM,SUBNET>>;
template <typename SUBNET> using level6 = res<2*FNUM,res<2*FNUM,res_down<2*FNUM,SUBNET>>>;

template <typename SUBNET> using alevel2 = ares_down<FNUM*5,SUBNET>;
template <typename SUBNET> using alevel3 = ares<FNUM*4,ares_down<FNUM*4,SUBNET>>;
template <typename SUBNET> using alevel4 = ares_down<8*FNUM,SUBNET>;
template <typename SUBNET> using alevel5 = ares<4*FNUM,ares_down<4*FNUM,SUBNET>>;
template <typename SUBNET> using alevel6 = ares<2*FNUM,ares<2*FNUM,ares_down<2*FNUM,SUBNET>>>;
// training network type
using net_type =    loss_multimulticlass_log<fc<56,fc<256,avg_pool_everything<
                            level5<
                            level6<
                            max_pool<3,3,2,2,relu<bn_con<con<FNUM,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type =   loss_multimulticlass_log<fc<56,fc<256,avg_pool_everything<
                            alevel5<
                            alevel6<
                            max_pool<3,3,2,2,relu<affine<con<FNUM,7,7,2,2,
                            input<matrix<float>>
                            >>>>>>>>>>;

#endif // CUSTOMNETWORK_H
