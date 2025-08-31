# Representative Snippet-aware Modeling Based on Points for Weakly-supervised Temporal Action Localization
## Prerequisites
### Recommended Environment
* Python 3.7
* Pytorch 1.8.1
* Tensorflow 1.15 (for Tensorboard)
* CUDA 11.1

### Depencencies
You can set up the environments by using `$ pip install -r requirements.txt`.

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.

2. we provide extracted features from the link provided in [this repo](https://github.com/ispc-lab/ACM-Net/blob/main/README.md#motivation).
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── point_gaussian
           └── point_labels.csv
       └── features
           ├── train
                ├── video_validation_0000051.npy
                ├── video_validation_0000052.npy
                └── ...
           └── test
                ├── video_test_0000004.npy
                ├── video_test_0000006.npy
                └── ...
             
~~~~

## Usage

### Running
You can easily train and evaluate the model by running the script below.

If you want to try other training options, please refer to `options.py`.

~~~~
$ bash train.sh
~~~~
#### Tip
Our model uses the second GPU for training. If the GPU order is to be modified, the files need to be changed.

### Evaulation
The pre-trained model can be found [here](https://drive.google.com/file/d/1yZKI6h5nz59yqS9K4eyusKh4nEyH79JQ/view?usp=share_link).
You can evaluate the model by running the command below.
~~~~
$ bash eval.sh
~~~~
