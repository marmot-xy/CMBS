## Cross Modal Background Suppression for Audio-Visual Event Localization


This is a pytorch implementation for CVPR 2022 paper "Cross Modal Background Suppression for Audio-Visual Event Localization"



## Data preparation
The VGG visual features can be downloaded from [Visual_feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing)

The VGG-like audio features can be downloaded from [Audio_feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing)

The noisy visual features used for weakly-supervised setting can be downloaded from [Noisy_visual_feature](https://drive.google.com/file/d/1I3OtOHJ8G1-v5G2dHIGCfevHQPn-QyLh/view?usp=sharing)

After downloading the features, please place them into the `data` folder.

##Pretrained model
The pretrained models can be downloaded from [Supervised model][Supervised_model] and [WeaklySupervised model][WeaklySupervised_model].

## Acknowledgement

Part of our code is borrowed from the following repositories.

- [YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18)
- [CMRAN](https://github.com/FloretCat/CMRAN)


We thank to the authors for releasing their codes. Please also consider citing their works.




[Supervised_model]: https://drive.google.com/file/d/1crF9vKpdi3Ec_Zkagz7rJB_yVHYnxpJE/view?usp=sharing

[WeaklySupervised_model]: https://drive.google.com/file/d/100cp82dIrJLuqqEvV-9dbxyTQpWkY-3c/view?usp=sharing
