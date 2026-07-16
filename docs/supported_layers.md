# Supported Lux layers
The following is a list of the supported and tested Lux layers. Several untested layers may work, and tested layers may still break when using symbolic dimensions.

* Containers
    - [x] BranchLayer
    - [x] Chain
    - [x] PairwiseFusion
    - [x] Parallel
    - [x] SkipConnection
    - [x] RepeatedLayer
    - [x] AlternatePrecision
* Convolutional Layers
    - [x] Conv
    - [ ] ConvTranspose
* Dropout Layers
    - [ ] AlphaDropout
    - [x] Dropout
    - [ ] VariationalHiddenDropout
* Pooling Layers
    - [ ] AdaptiveLPPool
    - [ ] AdaptiveMaxPool
    - [ ] AdaptiveMeanPool
    - [x] GlobalLPPool
    - [x] GlobalMaxPool
    - [x] GlobalMeanPool
    - [ ] LPPool
    - [ ] MaxPool
    - [ ] MeanPool
* Recurrent Layers
    - [ ] GRUCell
    - [ ] LSTMCell
    - [ ] RNNCell
    - [ ] Recurrence
    - [ ] StatefulRecurrentCell
    - [ ] BidirectionalRNN
* Linear Layers
    - [x] Bilinear
    - [x] Dense
    - [x] Scale
* Attention Layers
    - [x] MultiHeadAttention
* Embedding Layers
    - [ ] Embedding
    - [ ] RotaryPositionalEmbedding
    - [ ] SinusoidalPositionalEmbedding
* Functional API
    - [ ] apply_rotary_embedding
    - [ ] compute_rotary_embedding_params
* Misc. Helper Layers
    - [x] FlattenLayer
    - [x] Maxout
    - [x] NoOpLayer
    - [x] ReshapeLayer
    - [x] SelectDim
    - [x] WrappedFunction
    - [x] ReverseSequence
* Normalization Layers
    - [ ] BatchNorm
    - [ ] GroupNorm
    - [ ] InstanceNorm
    - [x] LayerNorm
    - [ ] WeightNorm
    - [ ] RMSNorm
* Upsampling
    - [ ] PixelShuffle
    - [ ] Upsample
