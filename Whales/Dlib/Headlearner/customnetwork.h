#ifndef CUSTOMNETWORK_H
#define CUSTOMNETWORK_H

#include <dlib/dnn.h>

namespace dlib {

// training network type
using net_type = loss_metric<fc_no_bias<512,relu<dropout<fc<1024,dropout<input<matrix<float>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<512,relu<multiply<fc<1024,multiply<input<matrix<float>>>>>>>>;
}

#endif // CUSTOMNETWORK_H
