# See Far in the Dark

PyTorch code for our ACCV 2024 paper `[Joint Image Super-resolution and Low-light Enhancement in the Dark](https://openaccess.thecvf.com/content/ACCV2024/html/Zhou_Joint_Image_Super-resolution_and_Low-light_Enhancement_in_the_Dark_ACCV_2024_paper.html)`

## Dataset

You can download this dataset from [Google Cloud Drive](https://drive.google.com/drive/folders/1i68Vs-l21UtOMIsVlXnVoxvFKVqrLyKb?usp=sharing) or [Baidu Cloud Drive](https://pan.baidu.com/s/1RN9p_WBPL6Sh8FU6Go8qvg?pwd=dark)(code:dark). And 1395 pairs of RAW and sRGB images for training, 153 for testing. 

Now the inference/test code and our DarkSR dataset are available, and the training code and experimental settings will come soon...

In general, you only need to download `DarkSR.rar` and `Real-DarkSR.rar`, the others are the images of a specific camera model classified from DarkSR.


## Training

You can make use of this code to run the network with three input types, just change the input_type parameter of config.py, which supports raw, srgb, dual, in order to reduce the reading time of raw images and jpg images, we save the image data in .npy format.


When training, please set the `input type`, `camera model` (the specific dataset used, DarkSR or a specific camera model), and the `experiment name` (suggested: date_dataset_network_name), and set `b_test_only` to False when training. there are some networks that have a competition for the size of the inputs, so in order to increase the runnability, it can be set to crop into chunks for feeding into the network , the output is then stitched together into a complete image, the image cropping size at the time of testing is 256,512,1024, etc., depending on the image resolution used.

## Testing

During the test phase, only `b_test_only` needs to be set to True, and whether or not to crop.

## Inference

![image](https://github.com/user-attachments/assets/d07207d4-e222-4d34-8593-9f672fcbdcd2)


You can also use your own captured raw+srgb data to perform inference using the trained network, first make sure that the files are placed correctly (reference/raw and reference/isp) and then run the Inference.py file with the output in reference/result.

It is worth noting that with RAW+JPG images taken with a camera, different camera manufacturers may have set a default cropping region, resulting in the actual JPG resolution being smaller than RAW, one lazy way is to render the JPG image using rawpy, but such a jpg is tonally different from that taken with a camera, and the recommended way to do it is to get rid of this cropping region. The resolution of a jpg taken by a mobile phone is generally equal to RAW and no extra care is needed.

If you find our repo useful for your research, please consider citing this paper and our previous work

```
@InProceedings{JSLNet,
    author    = {Zhou, Feihu and Chang, Kan and Ling, Mingyang and Li, Hengxin and Xia, Shucheng},
    title     = {Joint Image Super-resolution and Low-light Enhancement in the Dark},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {1332-1349}
}
```

```
@ARTICLE{PRNet,
    author={Ling, Mingyang and Chang, Kan and Huang, Mengyuan and Li, Hengxin and Dang, Shuping and Li Baoxin},
    journal={IEEE Transactions on Computational Imaging},
    title={PRNet: Pyramid Restoration Network for RAW Image Super-Resolution},
    year={2024},
    volume={10},
    pages={479-495},
    doi={10.1109/TCI.2024.3374084}
}
```


