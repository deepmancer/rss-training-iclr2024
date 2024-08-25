# Out-Of-Domain Unlabeled Data Improves Generalization - ICLR 2024 (Spotlight)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.00027)
[![poster](https://img.shields.io/badge/Poster-PDF-87CEEB)](https://iclr.cc/media/PosterPDFs/ICLR%202024/19202.png?t=1712876187.1666338)
[![presentation](https://img.shields.io/badge/Presentation-ICLR%202024-FFA500)](https://iclr.cc/virtual/2024/poster/19202)
[![openreview](https://img.shields.io/badge/OpenReview-Discussion-B762C1)](https://openreview.net/forum?id=Bo6GpQ3B9a)

> [**Out-Of-Domain Unlabeled Data Improves Generalization**](https://arxiv.org/abs/2310.00027) <be>
> [Seyed Amir Hossein Saberi](https://scholar.google.com/citations?user=OyvmpN4AAAAJ&hl=en),
> [Amir Najafi](https://scholar.google.com/citations?user=N_zYPC0AAAAJ&hl=en),
> [Alireza Heidari](https://www.linkedin.com/in/alireza-heidari-7b55721bb/),
> [Mohammad Hosein Movasaghinia](https://scholar.google.com/citations?user=68otW_4AAAAJ&hl=en),
> [Abolfazl Motahari](https://scholar.google.com/citations?user=rJ-biB0AAAAJ&hl=en),
> [Babak HosseinKhalaj](https://scholar.google.com/citations?user=8HsoXAUAAAAJ&hl=en)
<br>**Sharif University of Technology**<br>

Official implementation


<details>
    <summary>Click for Abstract</summary>
    We propose a **novel framework** for incorporating **unlabeled data** into semi-supervised classification problems, where scenarios involving the minimization of either:

- *i)* adversarially robust, or 
- *ii)* non-robust loss functions 

have been considered. Notably, we allow the unlabeled samples to deviate slightly (in the total variation sense) from the in-domain distribution. The core idea behind our framework is to combine **Distributionally Robust Optimization (DRO)** with **self-supervised training**. As a result, we also leverage **efficient polynomial-time algorithms** for the training stage.

From a theoretical standpoint, we apply our framework to the classification problem of a **mixture of two Gaussians** in $\mathbb{R}^d$, where, in addition to the $m$ independent and labeled samples from the true distribution, a set of $n$ (usually with $n \gg m$) out-of-domain and unlabeled samples are also provided.

Using only the labeled data, it is known that the generalization error can be bounded by:

$$\propto \left(\frac{d}{m}\right)^{1/2}.$$

However, using our method on both isotropic and non-isotropic Gaussian mixture models, one can derive a new set of **analytically explicit and non-asymptotic bounds** which show substantial improvement in the generalization error compared to ERM.

Our results underscore two significant insights:

1. Out-of-domain samples, even when unlabeled, can be harnessed to narrow the generalization gap, provided that the true data distribution adheres to a form of the "cluster assumption."
2. The semi-supervised learning paradigm can be regarded as a special case of our framework when there are no distributional shifts.

We validate our claims through experiments conducted on a variety of synthetic and real-world datasets.

</details>
