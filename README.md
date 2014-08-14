Music Autotagging demo
======

A demo code for performing music autotagging. The audio features of the example music dataset used in this demo (majorminer) are computed using the audio analysis library [essentia](http://essentia.upf.edu/)
The demo includes all the necessary code, metadata and audio features.
This demo is adapted from the Music Autotagging tutorial @ [ISMIR 2013](http://ismir2013.ismir.net/)

Dependencies
------

*[scikit-learn](http://scikit-learn.org/stable/install.html)

..*For Linux Ubuntu
....*sudo apt-get install build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev libatlas3-base
....*sudo pip install -U scikit-learn
#(for Ubuntu versions equal or below 11.10, libatlas3-base is called libatlas-base-dev)
*For other OS
....*Instructions to install it: [http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html)

*[pyyaml](http://pyyaml.org/)
..*For Linux/Mac OS X
....*sudo pip install pyyaml


Example of how to run this demo
------

* python feature_preprocessing.py majorminer --include-features lowlevel.* rhythm.* --remove-features lowlevel.mfcc.*
* python feature_selection.py majorminer
* python classification.py majorminer [-a all]
* python evaluation.py majorminer [-a all] 

[-a all] is optional, that's the default behavior, it will classify and evaluate all the "implemented" classifiers

Please have a look at the (very short) documentation on each one of the previous python scripts