# InnovR 2018/2019.

This repository contains the code for the InnovR project "On Automating Data Augmentation for Deep Image Denoising" done in the 2018/2019 academic year.

# Table of contents

## Data Module

Provides a common interface for dataset objects, which provide data to Deep Learning Keras-based models.

## Transformations Module

Uses [Imgaug](https://github.com/aleju/imgaug) to implement policy image transformations. Contain definitions about operations, subpolicies and policy objects.

## Models module

Implementation of a Denoising Autoencoder in Keras. Neural network architecture:

![](https://github.com/eddardd/Automatic-Data-Augmentation/blob/master/DAE_Architecture.png)

## Controller module

Contains the code for the Controller Network used to predict Optimal Augmentation policies

## Tests

Codes to generate paper figures, perform experiments and train controller network.
