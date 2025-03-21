# TANGAN

Welcome to the TANGAN Repository! This repository contains the supplementary material referenced in the paper "TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network".

🚀 Exciting News! 🚀

Our paper has been accepted for publication in [Applied Intelligence](https://doi.org/10.1007/s10489-025-06364-x)!

💡 Title: "TANGAN: Solving Tangram Puzzles Using Generative Adversarial Network"

📌 Authors: Fernanda Miyuki Yamada, Harlen Costa Batagelo, João Paulo Gois, Hiroki Takahashi

📖 Journal: Applied Intelligence (Springer)


## Abstract

While humans show remarkable proficiency in solving visual puzzles, machines often fall short due to the complex combinatorial nature of such tasks. Consequently, there is a growing interest in developing computational methods for the automatic solution of different puzzles, especially through deep learning approaches. The Tangram, an ancient Chinese puzzle, challenges players to arrange seven polygonal pieces to construct different patterns. Despite its apparent simplicity, solving the Tangram is considered an NP-complete problem, being a challenge even for the most sophisticated algorithms. Moreover, ensuring the generality and adaptability of machine learning models across different Tangram arrangements and complexities is an ongoing research problem. In this paper, we introduce a generative model specifically designed to solve the Tangram. Our model competes favorably with previous methods regarding accuracy while delivering fast inferences. It incorporates a novel loss function that integrates pixel-based information with geometric features, promoting a deeper understanding of the spatial relationships between pieces. Unlike previous approaches, our model takes advantage of the geometric properties of the Tangram to formulate a solving strategy, exploiting its inherent properties only through exposure to training data rather than through direct instruction. Extending the proposed loss function, we present a novel evaluation metric as a better fitting measure for assessing Tangram solutions than previous metrics. We further provide a new dataset containing more samples than others reported in the literature. Our findings highlight the potential of deep learning approaches in geometric problem domains.

## Proposed Architecture

TANGAN uses deep learning to solve Tangram puzzles by understanding the spatial relationships between pieces through training data exposure. This approach contrasts with previous methods that require direct instruction on geometric properties. The architecture and its components are illustrated in the image below:


![cover_image](https://github.com/fernandamyamada1/TANGAN/assets/20599223/1fb98c68-d633-42df-b7e6-8eadf9cb90ca)

The generator architecture for TANGAN is presented below: 

![generator](https://github.com/fernandamyamada1/TANGAN/blob/main/generator_arch.png)

The discriminator architecture for TANGAN is presented below: 

![discriminator](https://github.com/fernandamyamada1/TANGAN/blob/main/discriminator_arch.png)

The training code is available.

[Download Code](https://github.com/fernandamyamada1/TANGAN/blob/main/TANGAN.py)

A demonstration video shows the strategy used by TANGAN to solve different Tangram puzzles.

[Download Video](https://github.com/fernandamyamada1/TANGAN/tree/main/final_video.mov)



https://github.com/user-attachments/assets/7bdfc37c-2c60-4b7a-91f6-db4d037001cb



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

## Supplementary Studies

 In our supplementary studies, we evaluate the domain adaptability performance of TANGAN by testing its ability to solve other geometric puzzles: polyominoes and tesselations.

The dataset for supplementary studies is available. When running tests on these datasets, substitute the variable tones in the training code for the values in the tones.txt file. 

- [Download Supplementary Studies Dataset](https://github.com/fernandamyamada1/TANGAN/tree/main/supplementary/dataset/).
  
Generated solutions for the supplementary studies are also available.

- [Supplementary Studies Solutions](https://github.com/fernandamyamada1/TANGAN/tree/main/supplementary/solutions/).




## Getting Started

### Installation
Clone the repository.

```
git clone https://github.com/fernandamyamada1/TANGAN.git
cd TANGAN
```

### Prerequisites
We implemented TANGAN using Python 3.9. Ensure you have the required dependencies installed by referring to the list provided in the requirements.txt file.

```
pip install -r requirements.txt
```

## Citation


If you use this repository, please cite:


```bibtex
@article{yamada2025tangan,
  title={TANGAN: solving Tangram puzzles using generative adversarial network},
  author={Yamada, Fernanda Miyuki and Batagelo, Harlen Costa and Gois, Jo{\~a}o Paulo and Takahashi, Hiroki},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={1--27},
  year={2025},
  publisher={Springer}
}
