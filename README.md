# PAIR defense on text-based Image Retrieval

Our defense paper: PAIR: Pre-denosing Augmented Image Retrieval Model for Defending Adversarial Patches

## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```sh
conda create -n tth python==3.8 -y
conda activate tth
pip install -r requirements.txt
```



## Data prepare

### Dataset

Put the dataset files on `~/VisualSearch`.

```sh
mkdir ~/VisualSearch
unzip -q "TTH_VisualSearch.zip" -d "${HOME}/VisualSearch/"
```

Readers need to download Flickr30k dataset and move the image files to `~/VisualSearch/flickr30k/flickr30k-images/`.

Download the pretrained model. 

```sh
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
```



## Train defense model
```sh
python -u train_defense_model.py
```



## Test defense model on clean images
```sh
python -u test_defense_model_on_clean.py
```





## Pacth attack


```sh
 python TTH_attack.py \
 --device 0 flickr30ktest_add_ad None flickr30ktrain/flickr30kval/test \
 --attack_trainData flickr30ktrain --config_name TTH.CLIPEnd2End_adjust \
 --parm_adjust_config 0_1_1 --rootpath ~/VisualSearch \
 --batch_size 256 --query_sets flickr30ktest_add_ad.caption.txt
```

## Patch attack with our defense

You can select the keyword: jacket dress floor female motorcycle policeman cow waiter swimming reading run dancing floating smiling climbing feeding front little green yellow pink navy maroon.

```sh

 python -u attack_with_our_defense.py \
 --device 0 flickr30ktest_add_ad None flickr30ktrain/flickr30kval/test \
 --attack_trainData flickr30ktrain --config_name TTH.CLIPEnd2End_adjust \
 --parm_adjust_config 0_1_1 \
 --batch_size 256 --query_sets flickr30ktest_add_ad.caption.txt \ 
 --keyword jacket
```






