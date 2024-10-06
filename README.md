PyTorch code for our ACCV 2024 paper `Joint Image Super-resolution and Low-light Enhancement in the Dark`

## Dataset



You can download this dataset from [Google Cloud Drive](https://drive.google.com/drive/folders/1i68Vs-l21UtOMIsVlXnVoxvFKVqrLyKb?usp=sharing) or [Baidu Cloud Drive](https://drive.google.com/drive/folders/1i68Vs-l21UtOMIsVlXnVoxvFKVqrLyKb?usp=sharing). And 1395 pairs of RAW and sRGB images for training, 153 for testing. 

The training and testing code will come soon...



In general, you only need to download `DarkSR.rar` and `Real-DarkSR.rar`, the others are the images of a specific camera model classified from DarkSR.

## Training

You can make use of this code to run the network with three input types, just change the input_type parameter of config.py, which supports raw, srgb, dual, in order to reduce the reading time of raw images and jpg images, we save the image data in .npy format.


When training, please set the `input type`, `camera model` (the specific dataset used, DarkSR or a specific camera model), and the `experiment name` (suggested: date_dataset_network_name), and set `b_test_only` to False when training. there are some networks that have a competition for the size of the inputs, so in order to increase the runnability, it can be set to crop into chunks for feeding into the network , the output is then stitched together into a complete image, the image cropping size at the time of testing is 256,512,1024, etc., depending on the image resolution used.

## Testing

During the test phase, only `b_test_only` needs to be set to True, and whether or not to crop.

## Inference

You can also use your own captured raw+srgb data to perform inference using the trained network, first make sure that the files are placed correctly (reference/raw and reference/isp) and then run the Inference.py file with the output in reference/result.

It is worth noting that with RAW+JPG images taken with a camera, different camera manufacturers may have set a default cropping region, resulting in the actual JPG resolution being smaller than RAW, one lazy way is to render the JPG image using rawpy, but such a jpg is tonally different from that taken with a camera, and the recommended way to do it is to get rid of this cropping region. The resolution of a jpg taken by a mobile phone is generally equal to RAW and no extra care is needed.
