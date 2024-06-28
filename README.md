# TANGAN

Welcome to the Tangram Puzzle Dataset and Code Repository! This repository contains the supplementary material referenced in the technical paper "TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network" presented at SIGGRAPH Asia 2024.

# Dataset
The dataset included in this repository is the most extensive collection in the literature concerning the automated solution of Tangram puzzles. It consists of:

- 5,900 training samples
- 100 testing samples

The images are stored in grayscale with 512x512 pixels in size. The dataset includes patterns with holes, multiple regions, and unconstrained rotations.

# Networks

The Python scripts for the proposed TANGAN architecture is provided.

# Generated Solutions
The generative model in this repository provides solutions to Tangram puzzles using a novel loss function that integrates pixel-based information with geometric features. The model generates solutions that can be compared with those produced by the VAE-GAN architecture.

# Requirements
The code provided in this repository implements the TANGAN model using Python 3.9.12 and TensorFlow 2.11.0. The experimental setup includes:

Hardware: Ryzen 3700x 3.6GHz, 32GB RAM, Nvidia RTX 4090 24GB
Adam Optimizer with a learning rate of 0.0002 and first-moment decay rate of 0.5
Binary cross-entropy loss for the discriminator and a custom loss function for the generator
Training process divided into two stages with a batch size of 20
Requirements
The necessary Python packages and their versions are listed in the requirements.txt file for easy installation.
