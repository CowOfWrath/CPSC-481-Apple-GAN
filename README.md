# CPSC-481-Apple-GAN
Class project for Fall 2019 CPSC-481

Ryan Chen - 893219394 - ryan.chen@csu.fullerton.edu

Frank Ngo - 889272738 - frank.ngo@csu.fullerton.edu

Functional descriptions:
apple_gan.py contains to 2 models to generate and discriminate fake apple images, apply noise filter, and save images to disk.
GUI.py creates a GUI application for a mini game: player will guess whether the images displayed are real or fake. After the player selects their choice, a prompt will display the result and ask whether the player wants to play again or quit the application.

Run instructions:
The number of epochs can be modified via the variable EPOCHS in apple_gan.py
Our project is self-contained. No external library needs to be loaded before running the python files.
To train and generate fake apple images, run
python apple_gan.py
There are some warnings, but they can be discarded.
To run the GUI application, run
python GUI.py


