# EXPERIMENT SUGGESTIONS (more reasonable ideas first?)

* just train for longer
     - d.h5 never stopped improving, just slowed down
     - perhaps with increased learning rate (at least to start)
* normalize input greyscale, color hint, color output 
* sharper `hint_sample_variance` (4)?
     - I think we average the colors over too large an area and network gets undecisive
* leakyrelu
     - apparently good for deep networks to fix dead neurons
* use pretrained model
     - would be great 
     - but regular autoencoder/unet models have totally different input, output shape
         - find some way to make this work
     - using a pretrained colorizer feels gross
* remove low chroma images (and greyscale images) from existing set
* more sophisticated weight initialization (Colorful Colorization, uses k-means initialization from "Data-dependent initializations of convolutional neural networks.")
* new, bigger training image set
     - I think if we stick to training on a narrow class of images, we might get better results with the limited compute that we have
     - could just use a bigger flower dataset
* adam parameters other than learning rate
* investigate impact of `hint_radius`?
     - really shouldn't matter but somehow I think it does
     - pass normalized version of sample_flat's `sample` variable instead of hint_mask 
* am I wrong that shortcut connections are bad
     - I feel like we don't need shortcut connections because we don't need high output resolution
     - contribute to the network not utilizing its full depth
         - which is bad because something about receptive field / hints don't propogate
     - I feel like we just want a regular convolutional autoencoder
* downsample more than half each step
* smaller bottleneck (more downsampling steps)
* add more depth to the bottleneck
* various other ways to downsample (maxpool, averagepool)
* investigate other loss functions
     - GAN loss would be nice, difficult
     - current loss considers differences in output across pixels as the same as across channels
     - perhaps loss should treat colors as their own units / bin output color?
* batchnormalization vs groupnormalization (i hear group is better for small batches)
* first train model with lots of hints, then decrease hinted area over time
* hint point sampled from normal centered on image center (from paper)

# TRAINING SCHEDULE FROM Colorful Image Colorization

it uses adam

don't know batch size or what the number of iterations means

1.3M images from the ImageNet training set,
validate on the first 10k images in the ImageNet validation set, and test on a
separate 10k images in the validation set,

β1 = .9, β2 = .99, and weight decay = 10e−3 . Initial learning rate was 3 × 10e−5 and
dropped to 10e−5 and 3 × 10e−6 when loss plateaued, at 200k and 375k iterations,
respectively. 
