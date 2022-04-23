import tensorflow as tf
import tensorflow_text
import os

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
sep = os.path.sep
three_input_text = tf.constant([
    # This is my life.
    'Esta es mi vida.',
    # Are they still home?
    '¿Todavía están en casa?',
    # Try to find out.'
    'Tratar de descubrir.',
])
os.chdir(OR_PATH)
reloaded = tf.saved_model.load('translator')
result_es = reloaded.tf_translate(tf.constant(['Man riding surfboard']))

for tr in result_es['text']:
  print(tr.numpy().decode())

print()

result1 = reloaded.tf_translate(three_input_text )

for tr in result1['text']:
  print(tr.numpy().decode())

print()

