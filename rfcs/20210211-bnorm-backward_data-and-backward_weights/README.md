# Proposal for support backward data (actually not supported for scale and shift) and backward weights in batch normalization.

## Problem and motivation

Currently in OneDNN library batch normalization support following propagation 
kinds: forward_inference, forward_training, backward and backward_data (only 
without scale and shift). Tensorflow team requested the possibility to use 
backward_data with scale and shift. It is worth adding backward_weights 
propagation kind in this case as well. It will create full support for 
all possible propagation kinds in batch normalization.

## Current implementations

The current implementations are presented in the table below:


|                                                 | dnnl_forward_inference                                                  | dnnl_forward_training                                                                | dnnl_backward                                                                                     | dnnl_backward_data                                                                               |
| :--                                             | :--                                                                     | :--                                                                                  | :--                                                                                               | :--                                                                                              |
| dnnl_normalization_flags_none                   | *Inputs*: src <br><br> *Outputs*: dst                                   | *Inputs*: src <br><br> *Outputs*: dst, mu, sigma^2                                   | *Inputs*: diffdst, src, mu, sigma^2 <br><br> *Outputs*: diffsrc                                   | Same as for dnnl_backward                                                                        |
| dnnl_use_global_stats                           | *Inputs*: src, mu, sigma^2 <br><br> *Outputs*: dst                      | *Inputs*: src, mu, sigma^2 <br><br> *Outputs*: dst                                   | *Inputs*: diffdst, src, mu, sigma^2 <br><br> *Outputs*: diffsrc                                   | Same as for dnnl_backward                                                                        |
| dnnl_use_scaleshift                             | *Inputs*: src, gamma, beta <br><br> *Outputs*: dst                      | *Inputs*: src, gamma, beta <br><br> *Outputs*: dst, mu, sigma^2                      | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffsrc, diffgamma, diffbeta | Not supported                                                                                    |
| dnnl_use_global_stats \| dnnl_use_scaleshift    | *Inputs*: src, mu, sigma^2, gamma, beta <br><br> *Outputs*: dst         | *Inputs*: src, mu, sigma^2, gamma, beta <br><br> *Outputs*: dst                      | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffsrc, diffgamma, diffbeta | Not supported                                                                                    |
| `flags` \| dnnl_fuse_norm_relu                  | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags` | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags`, [Workspace] | *Inputs*: same as with `flags`, [Workspace] <br><br> *Outputs*: same as with `flags`              | Same as for dnnl_backward if `flags` do not contain dnnl_use_scaleshift; not supported otherwise |


## Proposal

### Option 1 (preferred option)

The first option is to add support for backward_data scale and 
backward_weights. The input and output data in such cases would look like this:

|                                              | dnnl_forward_inference                                                  | dnnl_forward_training                                                                | dnnl_backward                                                                                     | dnnl_backward_data                                                                                         | dnnl_backward_weights                                                                             |
| :--                                          | :--                                                                     | :--                                                                                  | :--                                                                                               | :--                                                                                                        | :--                                                                                               |
| dnnl_normalization_flags_none                | *Inputs*: src <br><br> *Outputs*: dst                                   | *Inputs*: src <br><br> *Outputs*: dst, mu, sigma^2                                   | *Inputs*: diffdst, src, mu, sigma^2 <br><br> *Outputs*: diffsrc                                   | Same as for dnnl_backward                                                                                  | Not supported                                                                                     |
| dnnl_use_global_stats                        | *Inputs*: src, mu, sigma^2 <br><br> *Outputs*: dst                      | *Inputs*: src, mu, sigma^2 <br><br> *Outputs*: dst                                   | *Inputs*: diffdst, src, mu, sigma^2 <br><br> *Outputs*: diffsrc                                   | Same as for dnnl_backward                                                                                  | Not supported                                                                                     |
| dnnl_use_scaleshift                          | *Inputs*: src, gamma, beta <br><br> *Outputs*: dst                      | *Inputs*: src, gamma, beta <br><br> *Outputs*: dst, mu, sigma^2                      | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffsrc, diffgamma, diffbeta | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta, diffgamma, diffbeta <br><br> *Outputs*: diffsrc <br><br> | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffgamma, diffbeta <br><br> |
| dnnl_use_global_stats \| dnnl_use_scaleshift | *Inputs*: src, mu, sigma^2, gamma, beta <br><br> *Outputs*: dst         | *Inputs*: src, mu, sigma^2, gamma, beta <br><br> *Outputs*: dst                      | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffsrc, diffgamma, diffbeta | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta, diffgamma, diffbeta <br><br> *Outputs*: diffsrc <br><br> | *Inputs*: diffdst, src, mu, sigma^2, gamma, beta <br><br> *Outputs*: diffgamma, diffbeta <br><br> |
| `flags` \| dnnl_fuse_norm_relu               | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags` | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags`, [Workspace] | *Inputs*: same as with `flags`, [Workspace] <br><br> *Outputs*: same as with `flags`              | *Inputs*: same as with `flags`, [Workspace] <br><br> *Outputs*: same as with `flags` <br><br>              | *Inputs*: same as with `flags`, [Workspace] <br><br> *Outputs*: same as with `flags` <br><br>     |

In case of batch normalization the weights are gamma and beta (scale and 
shift), so supporting backward_weights without use_scaleshift flag doesn't 
make sense. The behaviour of the individual cases can be summarized in the 
table below:

| propagation kind | behaviour                                             |
|------------------|-------------------------------------------------------|
| DW ss            | calculate data and weights, save scale and shift      |
| DW               | calculate data and weights, dont save scale and shift |
| W ss             | calculate weights save scale and shift                |
| D ss             | calculate data                                        |
| D                | calculate data and weights, dont save scale and shift |
| W                | Not supported                                         |

The implementation may just split the current one into two sections (calculate 
weights and calculate data) and execute it depending on propagation kind.

### Option 2
Option two is the same as the first except no support for backward_weights. 
Only the requested feature will be implemented (only additional support for 
backward_data with scale and shift which is actually unsupported). This will 
reduce the implementation effort but will not provide full user support.
