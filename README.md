# Final-Project-Group1: Image Captioning Model and Google Translation in TensorFlow
Coco Dataset
DATS 6203: Machine Learning 2 Final Group 1 Project at George Washington University (GWU) with Professor Dr. Amir Jafari. Included is the work on the class project with fellow teammates Adel Hassen (adelhassen@gwu.edu) and Xingyu (Alice) Yang (@gwu.edu). This code repository is based on the tutorial https://www.tensorflow.org/tutorials/text/image_captioning?msclkid=568d9785c23711ec9f4b4f0497afd299

You can find Image Captioning model implemented using TensorFlow in this repo. The model trains on MSCOCO data set which is downloadable from link: http://mscoco.org/dataset/#download

The model architecture is similar to Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.

This notebook is an end-to-end example. When you run the notebook, it downloads the MS-COCO dataset, preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

In this example, you will train a model on a relatively small amount of data—the first 30,000 captions for about 20,000 images (because there are multiple captions per image in the dataset).
