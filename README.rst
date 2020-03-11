======================================================================================
Task Decomposition and Synchronization Network (TDSNet)
======================================================================================

Semantic segmentation is essentially important to biomedical image analysis. We propose to decompose the single segmentation task into three subsequent sub-tasks, including (1) pixel-wise image segmentation, (2) prediction of the class labels of the objects within the image, and (3) classification of the scene the image belonging to. While these three sub-tasks are trained to optimize their individual loss functions of different perceptual levels, we propose to let them interact by the task-task context ensemble. Moreover, we propose a novel sync-regularization to penalize the deviation between the outputs of the pixel-wise segmentation and the class prediction tasks.We have successfully applied our framework to three diverse 2D/3D medical image datasets, including Robotic Scene Segmentation Challenge 18 (ROBOT18), Brain Tumor Segmentation Challenge 18 (BRATS18), and Retinal Fundus Glaucoma Challenge (REFUGE18). We have achieved top-tier performance in all three challenges. Our code, some part of data and trained models are available at this site.

.. contents::

Team members
------------
Xuhua Ren, Qian Wang and Dinggang Shen

Citation
----------

If you find this work useful for your publications, please consider citing::

  @article{ren2019task,
    title={Task Decomposition and Synchronization for Semantic Biomedical Image Segmentation},
    author={Ren, Xuhua and Zhang, Lichi and Ahmad, Sahar and Nie, Dong and Yang, Fan and Xiang, Lei and Wang, Qian and 
    Shen, Dinggang},
    journal={arXiv preprint arXiv:1905.08720},
    year={2019}
  }

Overview
--------
In general, we summarize our major contributions in this paper as follows:

1) We propose the task decomposition strategy to ease the challenging segmentation task in biomedical images.
2) We propose sync-regularization to coordinate the decomposed tasks, which gains advantage from multi-task learning toward image segmentation.
3) We build a practical framework for diverse biomedical image semantic segmentation and successfully apply it to three different challenge datasets. Our code is publicly available.

Data
----
`ROBOT18 <https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/>`_: The entire challenge dataset was made up of 19 sequences which were divided into 15 training sets and 4 test sets. Each sequence came from a single porcine training procedure recorded on da Vinci X or Xi system using specialized recording hardware. Sections of the procedure which contained significant camera motion or tissue interaction were extracted and subsampled to 1 Hz. Similar frames were manually removed until the sequence contained 300 frames. Each frame consists of a stereo pair with SXGA resolution 1280×1024 and intrinsic and extrinsic camera calibrations that were acquired during endoscope manufacture.

    .. figure:: images/fig1.PNG
        :scale: 100 %
        :align: center

`BRATS18 <https://www.med.upenn.edu/sbia/brats2018.html>`_: All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here. All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described in the BraTS reference paper, published in IEEE Transactions for Medical Imaging. The provided data are distributed after their pre-processing, i.e. co-registered to the same anatomical template, interpolated to the same resolution (1 mm) and skull-stripped.

    .. figure:: images/fig2.png
        :scale: 100 %
        :align: center
        
`REFUGE18 <https://refuge.grand-challenge.org/Details/>`_: A total of 1200 color fundus photographs are available. The dataset is split 1:1:1 into 3 subsets equally for training, offline validation and onsite test, stratified to have equal glaucoma presence percentage. Training set with a total of 400 color fundus image will be provided together with the corresponding glaucoma status and the unified manual pixel-wise annotations (a.k.a. ground truth). Testing consists of 800 color fundus images and is further split into 400 off-site validation set images and 400 on-site test set images.

    .. figure:: images/fig3.jpg
        :scale: 100 %
        :align: center

Method
------

Our proposed task decomposition framework as in figure. Given an input image (2D or 3D), we first use some convolutional operation to extract feature maps. Because of the diversity of the input 2D or 3D data, we design specific encoder for each of the three challenges in this paper. Then, we feed the extracted features to the task-task context ensemble module. The context ensemble module contains multi-scale dilated convolution, so the receptive fields are enlarged along the paths to combine features of different scales by different dilated rates. Moreover, the parallel context ensemble module is generated as task-task context ensemble module and each of module are connected by two branches which we called latent space. Finally, the network is decomposed from the latent space into three branches, corresponding to (1) the segmentation task, (2) the class task, and (3) the scene task. The decoders are trained for each decomposed task, including up sampling for the segmentation task, and also global average pooling, fully-connected layers and sigmoid or softmax activation function for the class and scene tasks. Note that the three decomposed tasks share the same latent space for decoding. Moreover, we adopt task synchronization to help the model training. Besides, such a regularize structure also suppresses the noises in the features from the shallow layers and produces accurate semantic predictions.

    .. figure:: images/fig4.png
        :scale: 100 %
        :align: center

Training
--------

1) Initialize the parameter (segmentation task) of the shared fully convolutional part using the pretrained net.
Initialize the parameters randomly from the normal distribution. 2) Based on 1), utilize SGD to train the segmentation-related net for updating these parameter and resulting. 3) Based on 1) and 2) utilize SGD to train the segmentation and class related net for updating these parameter. We train the segmentation and class tasks parameter together. 4) We utilize the early estimated parameters as initialization including 2) and 3), then refine the segmentation and class tasks with sync-regularization enforced with parameter. 5) We add in the scene task parameter and refine all of loss functions joint.

Results for ROBOT18
-------

Our method has demonstrated top-tier performance in the on-site testing set in ROBOT18 (rank second, IoU=61%, compared to 62% of the challenge winner), note that our method could outperformed others in intestine and kidney class. We also proposed a deep learning model testing strategy to combine a variety of input sizes, hyper-parameters of network in the segmentation task as ensemble inference framework which is our main contribution in our challenge paper. Meanwhile, removing small regions in segmentation map also adopted as post-processing module. The on-site test data IoU score is different with what is shown in Table, note that all the method details can be observed in challenge paper. Since the images in the on-site testing set and the validation set are not coherent with each other. We have also provided visual inspection of typical segmentation results of ROBOT18 in Fig including challenge winner solution OTH Regensberg, where our method clearly performs better than the alternatives under consideration.

    .. figure:: images/fig5.PNG
        :scale: 100 %
        :align: center

Here are some visualize results.

    .. figure:: images/fig6.png
        :scale: 100 %
        :align: center

Results for BRATS18
-------
TO DO

Results for REFUGE18
-------
TO DO

Dependencies
------------

* Anaconda with Python 3.7
* PyTorch 0.4.1
* TorchVision 0.2.1
* albumentations 0.0.4
* Pillow 6.1
* tqdm
* scikit-learn

To install all these dependencies, we have designed a script which could build all the requirement.
::

    #!/bin/bash
    bash init.sh



Run our code
----------

The dataset is organized in the folloing way:

::

    |-- cropted_train
    |   |-- seq_1
    |   |   |-- images
    |   |   |   |-- frame000.png
    |   |   |   |-- frame001.png
    |   |   |   |-- frame002.png
    |   |   |   |-- frame003.png
    |   |   |   |-- frame004.png
    |   |   |   |-- frame005.png
    |   |   |   `-- frame148.png
    |   |   `-- instruments_masks
    |   |       |-- frame000.png
    |   |       |-- frame001.png
    |   |       |-- frame002.png
    |   |       |-- frame003.png
    |   |       |-- frame004.png
    |   |       |-- frame005.png
    |   |       `-- frame148.png
    |-- raw
    |   |-- seq_1
    |   |   |-- labels
    |   |   |   |-- frame000.png
    |   |   |   |-- frame001.png
    |   |   |   |-- frame002.png
    |   |   |   |-- frame003.png
    |   |   |   |-- frame004.png
    |   |   |   `-- frame148.png
    |   |   |-- left_frames
    |   |   |   |-- frame000.png
    |   |   |   |-- frame001.png
    |   |   |   |-- frame002.png
    |   |   |   |-- frame003.png
    |   |   |   |-- frame004.png
    |   |   |   |-- frame005.png
    |   |   |   `-- frame148.png
    |   |   |-- right_frames
    |   |   |   |-- frame000.png
    |   |   |   |-- frame001.png
    |   |   |   |-- frame002.png
    |   |   |   |-- frame003.png
    |   |   |   |-- frame004.png
    |   |   |   |-- frame005.png
    |   |   |   `-- frame148.png
    |-- predictions

We have uploaded some ROBOT18 dataset in `Google Drive <https://drive.google.com/drive/folders/1bVleWfxUVXvCY6khhKEQWe8wiLip99KU?usp=sharing>`_. Considering these data not owned by us, we just upload a few images to help audience train and test code.

1. Preprocessing
~~~~~~~~~~~~~~~~~~~~~~
As a preprocessing step we cropped some image and transfer color label image to number one.

You can check ``python prepare.py --help`` will return set of all possible input parameters. And just run this bash file, the pre-processing work would be done.

::

    #!/bin/bash
    bash prepare.sh

2. Training
~~~~~~~~~~~~~~~~~~~~~~
The main file that is used to train all models ``train.py``.

Running ``python train.py --help`` will return set of all possible input parameters.

To train all models we used the folloing bash script :

::

    #!/bin/bash
    bash train.sh


3. Mask generation and Evaluation
~~~~~~~~~~~~~~~~~~~~~~
The main file to generate masks is ``test.py``.

Running ``python test.py --help`` will return set of all possible input parameters.

The evaluation is different for a multi-class segmentation: In the case of multi-class segmentation it calculates jaccard (dice) for every class independently then avaraged them for each image and then for every video.

To test all models we used the folloing bash script :

::

    #!/bin/bash
    bash test.sh


5. Further Improvements
~~~~~~~~~~~~~~~~~~~~~~

BRATS18 TODO

REFUGE18 TODO


