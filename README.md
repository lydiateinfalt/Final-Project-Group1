# Final-Project-Group1: Image Captioning Model and Google Translation in TensorFlow
DATS 6203: Machine Learning 2 Final Group 1 Project at George Washington University (GWU) with Professor Dr. Amir Jafari. Included is the work on the class project with fellow teammates Adel Hassen (adelhassen@gwu.edu) and Xingyu (Alice) Yang (yangxi47@gwu.edu). This code repository is based on a TensorFlow's Image Captioning [tutorial] <https://www.tensorflow.org/tutorials/text/image_captioning?msclkid=568d9785c23711ec9f4b4f0497afd299>

You can find Image Captioning model implemented using TensorFlow in this repo. The model trains on approximately 24000 images from COCO data set from 2014 which is downloadable from link: http://mscoco.org/dataset/#download

The model architecture is similar to Show, Attend and Tell: Neural Image Caption Generation with Visual Attention <https://arxiv.org/pdf/1502.03044.pdf?msclkid=97863852c49311ecb15903abb2f078b2>.


## Project Dependencies
  * TensorFlow Text 2.8.* (pip install 'tensorflow-text==2.8.*')
  * Google Translate 3.1.0a0 (pip install googletrans==3.1.0a0)
  * NLTK (pip install --user -U nltk)

The packages can be installed manually but the image_captioning_vgg16_BLEU.py does handle the installations automatically using os.system commands.

## Run Project Instructions
Navigate to Code directory and on a console with Python executables type in the following command "python3 image_captioning_vgg16_BLEU.py". 

This will class will download images from 2014 COCO dataset, build the model in TensorFlow, train the model using 24000 images and approximately 120,000 corresponding captions in 65 epochs and captions three New Yorker cartoons <https://github.com/nextml/caption-contest-data/tree/gh-pages/cartoons> in English. We use Google Translate API to translate the English caption to German, the output language can be specified to preference. The BLEU scores are displayed on the console when evaluating the generated captions in English and loss plot figure is saved to file. The entire class image_captioning_vgg16_BLEU.py takes approximately 45 minutes to run depending on hardware computational power.  
