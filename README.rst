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
        :scale: 80 %
        :align: center

`BRATS18 <https://www.med.upenn.edu/sbia/brats2018.html>`_: All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here. All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described in the BraTS reference paper, published in IEEE Transactions for Medical Imaging. The provided data are distributed after their pre-processing, i.e. co-registered to the same anatomical template, interpolated to the same resolution (1 mm) and skull-stripped.

    .. figure:: images/fig2.png
        :scale: 80 %
        :align: center
        
`REFUGE18 <https://refuge.grand-challenge.org/Details/>`_: A total of 1200 color fundus photographs are available. The dataset is split 1:1:1 into 3 subsets equally for training, offline validation and onsite test, stratified to have equal glaucoma presence percentage. Training set with a total of 400 color fundus image will be provided together with the corresponding glaucoma status and the unified manual pixel-wise annotations (a.k.a. ground truth). Testing consists of 800 color fundus images and is further split into 400 off-site validation set images and 400 on-site test set images.

    .. figure:: images/fig3.jpg
        :scale: 80 %
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
    |   |-- seq_16
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

The training dataset contains only 8 videos with 255 frames each. Inside each video all frames are correlated, so, for 4-fold cross validation of our experiments, we split data using this dependance i.e utilize whole video for the validation. In such a case, we try to make every fold to contain more or less equal number of instruments. The test dataset consists of 8x75-frame sequences containing footage sampled immediately after each training sequence and 2 full 300-frame sequences, sampled at the same rate as the training set. Under the terms of the challenge, participants should exclude the corresponding training set when evaluating on one of the 75-frame sequences. 

1. Preprocessing
~~~~~~~~~~~~~~~~~~~~~~
As a preprocessing step we cropped black unindormative border from all frames with a file ``prepare_data.py`` that creates folder ``data/cropped_train.py`` with masks and images of the smaller size that are used for training. Then, to split the dataset for 4-fold cross-validation one can use the file: ``prepare_train_val``.


2. Training
~~~~~~~~~~~~~~~~~~~~~~
The main file that is used to train all models -  ``train.py``.

Running ``python train.py --help`` will return set of all possible input parameters.

To train all models we used the folloing bash script :

::

    #!/bin/bash

    for i in 0 1 2 3
    do
       python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --type binary --jaccard-weight 1
       python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.00001 --n-epochs 20 --type binary --jaccard-weight 1
    done


3. Mask generation
~~~~~~~~~~~~~~~~~~~~~~
The main file to generate masks is ``generate_masks.py``.

Running ``python generate_masks.py --help`` will return set of all possible input parameters.

Example:
:: 
    python generate_masks.py --output_path predictions/unet16/binary --model_type UNet16 --problem_type binary --model_path data/models/unet16_binary_20 --fold -1 --batch-size 4

4. Evaluation
~~~~~~~~~~~~~~~~~~~~~~
The evaluation is different for a binary and multi-class segmentation: 

[a] In the case of binary segmentation it calculates jaccard (dice) per image / per video and then the predictions are avaraged. 

[b] In the case of multi-class segmentation it calculates jaccard (dice) for every class independently then avaraged them for each image and then for every video
::

    python evaluate.py --target_path predictions/unet16 --problem_type binary --train_path data/cropped_train

5. Further Improvements
~~~~~~~~~~~~~~~~~~~~~~

Our results can be improved further by few percentages using simple rules such as additional augmentation of train images and train the model for longer time. In addition, the cyclic learning rate or cosine annealing could be also applied. To do it one can use our pre-trained weights as initialization. To improve test prediction TTA technique could be used as well as averaging prediction from all folds.


6. Demo Example
~~~~~~~~~~~~~~~~~~~~~~
You can easily start working with our models using the demonstration example
  `Demo.ipynb`_

..  _`Demo.ipynb`: https://github.com/ternaus/robot-surgery-segmentation/blob/master/Demo.ipynb
.. _`Alexander Rakhlin`: https://www.linkedin.com/in/alrakhlin/
.. _`Alexey Shvets`: https://www.linkedin.com/in/shvetsiya/
.. _`Vladimir Iglovikov`: https://www.linkedin.com/in/iglovikov/
.. _`Alexandr A. Kalinin`: https://alxndrkalinin.github.io/
.. _`MICCAI 2017 Robotic Instrument Segmentation Sub-Challenge`: https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/
.. _`da Vinci Xi surgical system`: https://intuitivesurgical.com/products/da-vinci-xi/
.. _`TernausNet`: https://arxiv.org/abs/1801.05746
.. _`U-Net`: https://arxiv.org/abs/1505.04597
.. _`LinkNet`: https://arxiv.org/abs/1707.03718
.. _`Garcia`: https://arxiv.org/abs/1706.08126
.. _`Pakhomov`: https://arxiv.org/abs/1703.08580
.. _`google drive`: https://drive.google.com/open?id=13e0C4fAtJemjewYqxPtQHO6Xggk7lsKe

.. |br| raw:: html

   <br />

.. |plusmn| raw:: html

   &plusmn

.. |times| raw:: html

   &times

.. |micro| raw:: html

   &microm

.. |gif1| image:: images/original-min.gif
.. |gif2| image:: images/binary-min.gif
.. |gif3| image:: images/parts-min.gif
.. |gif4| image:: images/types-min.gif
.. |y| image:: images/y.gif
.. |y_hat| image:: images/y_hat.gif
.. |i| image:: images/i.gif