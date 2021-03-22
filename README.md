# Spoken Command Recognition

_This project was a speech processing course exercise in my university._
It uses HMM (hidden markov model) to recognize speech.

### Tested with Python 3.7.6

## How to use:

1.First you need some voice samples with `.wav` format for every word you want to recognize. Create a folder
named `dataset` (you can change folder names in constants) in the project root directory. Create a folder for each word
and put your samples inside that folder. For instance if you have two words like `music` and `telegram`, then you must
create a folder within `dataset` folder and put your voice samples inside folders named
`music` and `telegram`.

2.Now you need to run `train.py` to start the training process. It will look for folders in `dataset` and create a model
for each word inside another folder called `models`. Under the hood, it will extract features of every sample using
mfcc, mfcc delta, mfcc delta-delta, zero cross rating and energy.

3.Create a folder named `recognize`. Finally, you can run `recognize.py` to
start recognition process. Just copy your voice sample inside the folder.
