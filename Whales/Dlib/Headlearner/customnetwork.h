#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

namespace dlib {

// training network type
using net_type = loss_metric<fc_no_bias<512,
                             relu<bn_fc<fc<512,
                             relu<bn_fc<fc<512,
                             input<matrix<float>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<512,
                              relu<affine<fc<512,
                              relu<affine<fc<512,
                              input<matrix<float>>>>>>>>>>;
}

#endif // CUSTOMNETWORK_H
