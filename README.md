# CPSC-481-Apple-GAN
Class project for Fall 2019 CPSC-481

Ryan Chen - 893219394 - ryan.chen@csu.fullerton.edu

Frank Ngo - 889272738 - frank.ngo@csu.fullerton.edu


The dataset used for the project is a subset of images taken from: 

https://github.com/Horea94/Fruit-Images-Dataset
Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

The project utilizes portions of code from the following tutorials:

https://www.tensorflow.org/tutorials/generative/dcgan

https://www.tensorflow.org/tutorials/load_data/images


Functional descriptions:

apple_gan.py contains code to create two neural networks. One network generates images of apples and the other network discriminates between real and generated apple images. It also contains code to train the networks, save training data into checkpoints, and generate images and gifs to visualize the training progress.
The directory Apple contains the training dataset used to train the discriminator.
The directory training_checkpoints contains the training checkpoints and is accessed for loading and saving checkpoints.


GUI.py creates a GUI application for a mini game: player will guess whether the images displayed are real or fake. After the player selects their choice, a prompt will display the result and ask whether the player wants to play again or quit the application.
The directory Grayscales contains the images used by GUI.py


Run instructions:

apple_gan.py is dependent on the following libraries:__future__, tensorflow, glob, imageio, mathplotlib, numpy, os, PIL, IPython, and time

apple_gan.py is set up to train the two models after loading up the most recent checkpoint.
Before running the program, in the source code please edit lines 197 and 198 (the variables last_epoch and EPOCHS) to specify
the last training epoch used and the number of epochs to run. last_epoch is used to keep the file naming consistent when stopping and
starting a new training session. If last_epoch is not correctly set, the program will overwrite the files with newly generated images.


To train and generate fake apple images, run
python apple_gan.py
There are some warnings, but they can be discarded.


To run the GUI application, run
python GUI.py


