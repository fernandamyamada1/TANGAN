# TANGAN

Welcome to the TANGAN Repository! This repository contains the supplementary material referenced in the paper "TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network".

## Abstract

While humans show remarkable proficiency in solving visual puzzles, machines often fall short due to the complex combinatorial nature of such tasks. Consequently, there is a growing interest in developing computational methods for the automatic solution of different puzzles, especially through deep learning approaches. The Tangram, an ancient Chinese puzzle, challenges players to arrange seven polygonal pieces to construct different patterns. Despite its apparent simplicity, solving the Tangram is considered an NP-complete problem, being a challenge even for the most sophisticated algorithms. Moreover, ensuring the generality and adaptability of machine learning models across different Tangram arrangements and complexities is an ongoing research problem. In this paper, we introduce a generative model specifically designed to solve the Tangram. Our model competes favorably with previous methods regarding accuracy while delivering fast inferences. It incorporates a novel loss function that integrates pixel-based information with geometric features, promoting a deeper understanding of the spatial relationships between pieces. Unlike previous approaches, our model takes advantage of the geometric properties of the Tangram to formulate a solving strategy, leveraging its inherent properties only through exposure to training data rather than through direct instruction. Extending the proposed loss function, we present a novel evaluation metric as a better fitting measure for assessing Tangram solutions than other metrics proposed in the literature. We further provide a new dataset containing more samples than others reported in the literature. Our findings highlight the potential of deep learning approaches in geometric problem domains.

## Proposed Architecture

TANGAN uses deep learning to solve Tangram puzzles by understanding the spatial relationships between pieces through training data exposure. This approach contrasts with previous methods that require direct instruction on geometric properties. The architecture and its components are illustrated in the image below:


![cover_image](https://github.com/fernandamyamada1/TANGAN/assets/20599223/1fb98c68-d633-42df-b7e6-8eadf9cb90ca)

A demonstration video shows the strategy used by TANGAN to solve different Tangram puzzles:

[Watch Video](https://github.com/fernandamyamada1/TANGAN/tree/main/final_video.mov)



## Dataset
The dataset included in this repository is the most extensive collection in the literature concerning the automated solution of Tangram puzzles. It consists of:

- 5,900 training samples
- 100 testing samples

The images are stored in grayscale with 512x512 pixels in size. The dataset includes patterns with holes, multiple regions, and unconstrained rotations.

[Download Dataset](https://github.com/fernandamyamada1/TANGAN/tree/main/dataset/)


## Generated Solutions

This repository includes generated solutions using TANGAN and the VAE-GAN approach proposed by [Yamada et al. 2024] for comparison.

- [TANGAN Solutions](https://github.com/fernandamyamada1/TANGAN/tree/main/solutions/tangan)
- [VAE-GAN Solutions](https://github.com/fernandamyamada1/TANGAN/tree/main/solutions/vaegan)


## Getting Started

### Prerequisites
Ensure you have the required dependencies installed by referring to the list provided in the requirements.txt file.

```
pip install -r requirements.txt
```
### Installation
Clone the repository.

```
git clone https://github.com/fernandamyamada1/TANGAN.git
cd TANGAN
```




