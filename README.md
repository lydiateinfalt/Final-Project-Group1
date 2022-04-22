# Final-Project-Group1: Image Captioning Model and Google Translation in TensorFlow
Coco Dataset
DATS 6203: Machine Learning 2 Final Group 1 Project at George Washington University (GWU) with Professor Dr. Amir Jafari. Included is the work on the class project with fellow teammates Adel Hassen (adelhassen@gwu.edu) and Xingyu (Alice) Yang (yangxi47@gwu.edu). This code repository is based on a TensorFlow's Image Captioning tutorial [https://www.tensorflow.org/tutorials/text/image_captioning?msclkid=568d9785c23711ec9f4b4f0497afd299]

You can find Image Captioning model implemented using TensorFlow in this repo. The model trains on MSCOCO data set which is downloadable from link: http://mscoco.org/dataset/#download

The model architecture is similar to Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.

This notebook is an end-to-end example. When you run the notebook, it downloads the MS-COCO dataset, preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

In this example, you will train a model on a relatively small amount of data—the first 30,000 captions for about 20,000 images (because there are multiple captions per image in the dataset).

## Project Depencies
TensorFlow Text 2.8.* (pip install 'tensorflow-text==2.8.*')
Google Translate 3.1.0a0 (pip install googletrans==3.1.0a0)

The packages can be installed manually but the image_captioning_vgg16.py does handle the installations automatically using os.system commands.
