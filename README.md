# TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network

Welcome to the TANGAN Repository! This repository contains the supplementary material referenced in the technical paper "TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network" presented at SIGGRAPH Asia 2024.

## Abstract

While humans show remarkable proficiency in solving visual puzzles, machines often fall short due to the complex combinatorial nature of such tasks. Consequently, there is a growing interest in developing computational methods for the automatic solution of different puzzles, especially through deep learning approaches. The Tangram, an ancient Chinese puzzle, challenges players to arrange seven polygonal pieces to construct different patterns. Despite its apparent simplicity, solving the Tangram is considered an NP-complete problem, being a challenge even for the most sophisticated algorithms. Moreover, ensuring the generality and adaptability of machine learning models across different Tangram arrangements and complexities is an ongoing research problem. In this paper, we introduce a generative model specifically designed to solve the Tangram. Our model competes favorably with previous methods regarding accuracy while delivering fast inferences. It incorporates a novel loss function that integrates pixel-based information with geometric features, promoting a deeper understanding of the spatial relationships between pieces. Unlike previous approaches, our model takes advantage of the geometric properties of the Tangram to formulate a solving strategy, leveraging its inherent properties only through exposure to training data rather than through direct instruction. Our findings highlight the potential of deep learning approaches in geometric problem domains. Additionally, we provide a new dataset containing more samples than others reported in the literature.

## Proposed Architecture

TANGAN leverages deep learning to solve Tangram puzzles by understanding the spatial relationships between pieces through training data exposure. This approach contrasts with previous methods that require direct instruction on geometric properties. The architecture and its components are illustrated in the image below:

![TANGAN Architecture](path/to/your/image.png)


## Dataset
The dataset included in this repository is the most extensive collection in the literature concerning the automated solution of Tangram puzzles. It consists of:

- 5,900 training samples
- 100 testing samples

The images are stored in grayscale with 512x512 pixels in size. The dataset includes patterns with holes, multiple regions, and unconstrained rotations.

[Download Dataset](link/to/your/dataset)


## Generated Solutions

The `Solutions` folder includes generated solutions using TANGAN and the VAE-GAN approach proposed by [Yamada et al. 2024].

- [TANGAN Solutions](path/to/tangan/solutions)
- [VAE-GAN Solutions](path/to/vae-gan/solutions)

## Requirements
The code provided in this repository implements the TANGAN model using Python 3.9.12 and TensorFlow 2.11.0. The experimental setup includes:

Hardware: Ryzen 3700x 3.6GHz, 32GB RAM, Nvidia RTX 4090 24GB
Adam Optimizer with a learning rate of 0.0002 and first-moment decay rate of 0.5
Binary cross-entropy loss for the discriminator and a custom loss function for the generator
Training process divided into two stages with a batch size of 20
Requirements
The necessary Python packages and their versions are listed in the requirements.txt file for easy installation.


## Usage

To use this repository, clone it to your local machine and follow the instructions:

```bash
git clone https://github.com/yourusername/tangan.git
cd tangan

## Instalation

Install the required dependencies:

```bash
pip install -r requirements.txt



