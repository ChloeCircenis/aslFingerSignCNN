# ASL Finger Sign Classifier CNN 
Chloe Circenis 

University of Colorado Boulder, December 2025

## Project Overview
This project trains a CNN to recognize and classify images of ASL finger signs (letters). This model architecture could be expanded upon to build a search function for an ASL online dictionary or for a learning ASL app. The model is built in Python and utilizes Tensorflow, Keras, NumPy, Pandas, and Seaborn Libraries for data preprocessing, model architecture, and model evaluation. The model achieves a test accuracy of 56% overall with 50 epochs of training and thus could benefit from finetuning to achieve greater accuracy. 

## Index
- [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
- [Model Structure](#model-structure)
- [Model Evaluation](#model-evaluation)

## Dataset <a name="dataset"></a>
This project utilized Marxulia's [asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) Hugging Face dataset. This dataset contains 10873 pictures of ASL finger signs, approximately evenly distributed among the 26 finger signs (about 425 examples of each sign). 

### Data Preprocessing <a name="data-preprocessing"></a>
To prepare the data for the model, I utilized numpy and pandas to resize the images to be 64x64 and greyscale. I used one-hot encoding for the labels (A, B,..., Z). Then, once the data was in the ideal form for the model, I split it into training (80%) and test (20%) datasets and converted them to Tensorflow datasets. Then I created an additional validation (10%) set from the training set in order to cross validate the accuracy at each epoch when fitting the model. 

## Model Structure <a name="model-structure"></a>
The model has 3 convolutional layers with ReLU (rectified linear unit) activation functions. Each layer is interposed by a max pooling layer, and there is a final fully connected network to perform the final output classification. See full model architecture below: 


| Layer (type)                   | Output Shape         | Param #     |
|--------------------------------|-----------------------|-------------|
| **input_layer (InputLayer)**   | (None, 64, 64, 1)     | 0           |
| **conv2d (Conv2D)**            | (None, 64, 64, 32)    | 320         |
| **batch_normalization**        | (None, 64, 64, 32)    | 128         |
| **max_pooling2d**              | (None, 32, 32, 32)    | 0           |
| **conv2d_1 (Conv2D)**          | (None, 32, 32, 64)    | 18,496      |
| **batch_normalization_1**      | (None, 32, 32, 64)    | 256         |
| **max_pooling2d_1**            | (None, 16, 16, 64)    | 0           |
| **conv2d_2 (Conv2D)**          | (None, 16, 16, 128)   | 73,856      |
| **batch_normalization_2**      | (None, 16, 16, 128)   | 512         |
| **max_pooling2d_2**            | (None, 8, 8, 128)     | 0           |
| **flatten (Flatten)**          | (None, 8192)          | 0           |
| **dropout**                    | (None, 8192)          | 0           |
| **dense (Dense)**              | (None, 256)           | 2,097,408   |
| **batch_normalization_3**      | (None, 256)           | 1,024       |
| **dropout_1**                  | (None, 256)           | 0           |
| **dense_1 (Dense)**            | (None, 26)            | 6,682       |

Total params: 2,198,682 (8.39 MB)

Trainable params: 2,197,722 (8.38 MB)
 
Non-trainable params: 960 (3.75 KB)

## Model Evaluation <a name="model-evaluation"></a>

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/c6d0b34b-87b7-437a-84c7-05d7a971f5a6" />

While the overall test accuracy was around 56%, looking at the confusion matrix, the model actually performs well overall, but struggles with a few specific signs while excelling on others. The table below breaks down the scores for each sign individually. Where the model performed well, the letter is colored green, where it performed alright, the letter is colored yellow, and where is performed poorly, the letter is colored red. The color cutoffs chosen were 70% and 40%. 

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| $${\color{lightgreen}A}$$ | 0.7791 | 0.7791 | 0.7791 | 86 |
| $${\color{lightgreen}B}$$ | 0.7097 | 0.6947 | 0.7021 | 95 |
| $${\color{yellow}C}$$ | 0.5000 | 0.4359 | 0.4658 | 78 |
| $${\color{lightgreen}D}$$ | 0.7528 | 0.8272 | 0.7882 | 81 |
| $${\color{yellow}E}$$ | 0.5797 | 0.5128 | 0.5442 | 78 |
| $${\color{yellow}F}$$ | 0.5795 | 0.5795 | 0.5795 | 88 |
| $${\color{yellow}G}$$ | 0.6375 | 0.7083 | 0.6711 | 72 |
| $${\color{lightgreen}H}$$ | 0.7717 | 0.7320 | 0.7513 | 97 |
| $${\color{red}I}$$ | 0.4930 | 0.4023 | 0.4430 | 87 |
| $${\color{yellow}J}$$ | 0.6989 | 0.7386 | 0.7182 | 88 |
| $${\color{red}K}$$ | 0.3721 | 0.3636 | 0.3678 | 88 |
| $${\color{lightgreen}L}$$ | 0.7273 | 0.6914 | 0.7089 | 81 |
| $${\color{yellow}M}$$ | 0.5747 | 0.6024 | 0.5882 | 83 |
| $${\color{red}N}$$ | 0.4040 | 0.4762 | 0.4372 | 84 |
| $${\color{yellow}O}$$ | 0.6914 | 0.7467 | 0.7179 | 75 |
| $${\color{yellow}P}$$ | 0.6667 | 0.6429 | 0.6545 | 84 |
| $${\color{yellow}Q}$$ | 0.4857 | 0.4096 | 0.4444 | 83 |
| $${\color{red}R}$$ | 0.3563 | 0.3974 | 0.3758 | 78 |
| $${\color{yellow}S}$$ | 0.6250 | 0.6494 | 0.6369 | 77 |
| $${\color{red}T}$$ | 0.4138 | 0.4390 | 0.4260 | 82 |
| $${\color{red}U}$$ | 0.3297 | 0.3371 | 0.3333 | 89 |
| $${\color{red}V}$$ | 0.4353 | 0.4405 | 0.4379 | 84 |
| $${\color{yellow}W}$$ | 0.6098 | 0.5208 | 0.5618 | 96 |
| $${\color{red}X}$$ | 0.4545 | 0.5063 | 0.4790 | 79 |
| $${\color{red}Y}$$ | 0.3953 | 0.5075 | 0.4444 | 67 |
| $${\color{yellow}Z}$$ | 0.5063 | 0.4211 | 0.4598 | 95 |


