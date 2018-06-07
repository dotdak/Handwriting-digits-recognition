# Handwriting-digits-recognition

Firstly, install Anaconda

$ wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

$ bash ./Anaconda3-5.2.0-Linux-x86_64.sh

Then, install python on Anaconda

$ conda install -c anaconda python

Then, install Tensorflow on Python and all others libraries that listed in code. It is recommended to install Tensorflow in a virtual environment so that it doesn't harm to your Python software

$ conda create -n tensorflow pip python=3.6 # or python=2.7, etc.

$ source activate tensorflow (run this each time you want to use tensorflow)

$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl


How to run: 

$ python createmodel.py

  train and save in model2.ckpt
  
$ python predictmodel.py test_image.png

  feed image and export result
