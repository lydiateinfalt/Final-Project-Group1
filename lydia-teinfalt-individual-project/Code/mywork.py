
import tensorflow_text
from googletrans import Translator
os.system("sudo pip install 'tensorflow-text==2.8.*'")
os.system("sudo pip install googletrans==3.1.0a0")

translator = Translator()
translated_caption = ""

train_image_paths = image_paths[:10000]
print(len(train_image_paths))

# Added this to plot examples from the dataset
fig = plt.figure(figsize=(8,8))
i = 1
for j in range(6):
  n = np.random.randint(0, 100)
  ax1 = fig.add_subplot(2,3,i)
  plt.imshow(Image.open(img_name_vector[n]))
  print(train_captions[n])
  i += 1
  plt.savefig("sample_data.pdf")

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(224, 224)(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, image_path


image_model = tf.keras.applications.VGG16(include_top=False,
                                          weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
image_features_extract_model.summary()
plt.savefig('loss_plot.pdf')
plt.savefig("attention_plot.pdf")
"""## Try it on your own images

For fun, below you're provided a method you can use to caption your own images with the model you've just trained. Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for weird results!)

"""
image_url0 = 'https://raw.githubusercontent.com/nextml/caption-contest-data/gh-pages/cartoons/667.jpg'
image_url1 = 'https://raw.githubusercontent.com/nextml/caption-contest-data/gh-pages/cartoons/668.jpg'
image_url2 = 'https://raw.githubusercontent.com/nextml/caption-contest-data/gh-pages/cartoons/669.jpg'
image_list = [image_url0, image_url1, image_url2]

for image_url in image_list:
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)
    result, attention_plot = evaluate(image_path)
    plot_attention(image_path, result, attention_plot)
    # opening the image
    Image.open(image_path)
