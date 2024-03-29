#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

#define IMG_WIDTH  120
#define IMG_HEIGHT 120

#define FNUM 16

namespace dlib {

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

template <typename SUBNET> using level2 = res_down<4*FNUM,SUBNET>;
template <typename SUBNET> using level3 = res<2*FNUM,res_down<2*FNUM,SUBNET>>;
template <typename SUBNET> using level4 = res<FNUM,res<FNUM,SUBNET>>;

template <typename SUBNET> using alevel2 = ares_down<4*FNUM,SUBNET>;
template <typename SUBNET> using alevel3 = ares<2*FNUM,ares_down<2*FNUM,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<FNUM,ares<FNUM,SUBNET>>;


// training network type
using net_type = loss_multiclass_log<fc_no_bias<2,avg_pool_everything<
                                        level2<
                                        level3<
                                        level4<
                                        avg_pool<3,3,2,2,relu<bn_con<con<FNUM,7,7,2,2,
                                        input_rgb_image
                                        >>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc_no_bias<2,avg_pool_everything<
                                        alevel2<
                                        alevel3<
                                        alevel4<
                                        avg_pool<3,3,2,2,relu<affine<con<FNUM,7,7,2,2,
                                        input_rgb_image
                                        >>>>>>>>>>;
}

#endif // CUSTOMNETWORK_H
