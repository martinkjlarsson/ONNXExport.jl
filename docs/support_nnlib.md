# Supported NNlib functions

## Activation Functions
- [x] NNlib.celu
- [x] NNlib.elu
- [x] NNlib.gelu
- [x] NNlib.gelu_tanh
- [x] NNlib.gelu_sigmoid
- [x] NNlib.gelu_erf
- [x] NNlib.hardsigmoid
- [x] NNlib.sigmoid_fast
- [x] NNlib.hardtanh
- [x] NNlib.tanh_fast
- [x] NNlib.leakyrelu
- [x] NNlib.lisht
- [x] NNlib.logcosh
- [x] NNlib.logsigmoid
- [x] NNlib.mish
- [x] NNlib.relu
- [x] NNlib.relu6
- [x] NNlib.rrelu
- [x] NNlib.selu
- [x] NNlib.sigmoid
- [x] NNlib.softplus
- [x] NNlib.softshrink
- [x] NNlib.softsign
- [x] NNlib.swish
- [x] NNlib.hardswish
- [x] NNlib.tanhshrink
- [x] NNlib.trelu

## Attention
- [ ] NNlib.dot_product_attention
- [ ] NNlib.dot_product_attention_scores
- [ ] NNlib.make_causal_mask

## Softmax
- [x] NNlib.softmax
- [ ] NNlib.softmax!
- [x] NNlib.logsoftmax
- [ ] NNlib.logsoftmax!

## Pooling
- [x] NNlib.maxpool
- [x] NNlib.meanpool
- [x] NNlib.lpnormpool

## Padding
- [ ] NNlib.pad_reflect
- [ ] NNlib.pad_symmetric
- [ ] NNlib.pad_circular
- [ ] NNlib.pad_repeat
- [ ] NNlib.pad_constant
- [ ] NNlib.pad_zeros

## Convolution
- [ ] NNlib.conv
- [ ] NNlib.depthwiseconv
- [ ] NNlib.unfold
- [ ] NNlib.fold

## Upsampling
- [x] NNlib.upsample_nearest
- [x] NNlib.upsample_linear
- [x] NNlib.upsample_bilinear
- [x] NNlib.upsample_trilinear
- [ ] NNlib.pixel_shuffle

## Rotation
- [ ] NNlib.imrotate

## Batched Operations
- [x] NNlib.batched_mul
- [ ] NNlib.batched_mul!
- [ ] NNlib.batched_adjoint
- [ ] NNlib.batched_transpose
- [ ] NNlib.batched_vec

## Gather and Scatter
- [ ] NNlib.gather
- [ ] NNlib.gather!
- [ ] NNlib.scatter
- [ ] NNlib.scatter!

## Sampling
- [ ] NNlib.grid_sample

## Losses
- [ ] NNlib.ctc_loss

## Miscellaneous
- [x] NNlib.logsumexp
- [x] NNlib.glu
- [ ] NNlib.within_gradient
- [ ] NNlib.bias_act!
