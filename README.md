# ldvdGAN: Lower Dimensional Kernels for Video Discriminators

This is the implementation of the paper [Lower Dimensional Kernels for Video Discriminators ](https://arxiv.org/abs/1912.08860) based on [MoCoGAN](https://arxiv.org/abs/1707.04993).

Along adding ldvd models to the code, I have fixed TF2, scipy and python3 compatibility problems but some parts still needs improvement.

To use the Factorized video disciminator in the paper use:

`
    --video_discriminator FactorizedPatchVideoDiscriminator
`

option when specifying the video disciminator.

### still missing features / other changes to original code:
- This implementation does not reduce number of parameters as much as the original paper claims.  
- Time shifting is still not implemented
- Frechet Inception distance and Inception score should be evaluated
- Saving images, videos have been changed and might be sub-optimal.

For more details on training and generating videos, refer to the [MoCoGAN repo](https://github.com/sergeytulyakov/mocogan).
