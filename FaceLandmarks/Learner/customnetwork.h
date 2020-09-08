#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

#define IMG_WIDTH 100
#define IMG_HEIGHT 100

#define FNUM 8

using namespace dlib;

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

template <typename SUBNET> using level1 = res<8*FNUM,res_down<16*FNUM,SUBNET>>;
template <typename SUBNET> using level2 = res<8*FNUM,res_down<8*FNUM,SUBNET>>;
template <typename SUBNET> using level3 = res<4*FNUM,res_down<4*FNUM,SUBNET>>;
template <typename SUBNET> using level4 = res<2*FNUM,res<2*FNUM,SUBNET>>;

template <typename SUBNET> using alevel1 = ares<8*FNUM,ares_down<16*FNUM,SUBNET>>;
template <typename SUBNET> using alevel2 = ares<8*FNUM,ares_down<8*FNUM,SUBNET>>;
template <typename SUBNET> using alevel3 = ares<4*FNUM,ares_down<4*FNUM,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<2*FNUM,ares<2*FNUM,SUBNET>>;

// training network type
using net_type = loss_mean_squared_multioutput<fc_no_bias<136,avg_pool_everything<
                                        level1<
                                        level2<
                                        level3<
                                        level4<
                                        relu<bn_con<con<FNUM,5,5,2,2,
                                        input_rgb_image >>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_mean_squared_multioutput<fc_no_bias<136,avg_pool_everything<
                                         alevel1<
                                         alevel2<
                                         alevel3<
                                         alevel4<
                                         relu<affine<con<FNUM,5,5,2,2,
                                         input_rgb_image >>>>>>>>>>;

#endif // CUSTOMNETWORK_H
