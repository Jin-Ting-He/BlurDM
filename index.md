![Pipeline](assets/BlurDM_teaser.png)

## Abstract
Diffusion models show promise for dynamic scene deblurring; however, existing studies often fail to leverage the intrinsic nature of the blurring process within diffusion models, limiting their full potential. To address it, we present a Blur Diffusion Model (BlurDM), which seamlessly integrates the blur formation process into diffusion for image deblurring. Observing that motion blur stems from continuous exposure, BlurDM implicitly models the blur formation process through a dual-diffusion forward scheme, diffusing both noise and blur onto a sharp image. During the reverse generation process, we derive a dual denoising and deblurring formulation, enabling BlurDM to recover the sharp image by simultaneously denoising and deblurring, given pure Gaussian noise conditioned on the blurred image as input. Additionally, to efficiently integrate BlurDM into deblurring networks, we perform BlurDM in the latent space, forming a flexible prior generation network for deblurring. Extensive experiments demonstrate that BlurDM significantly and consistently enhances existing deblurring methods on four benchmark datasets.

## Results
![Vizs](assets/BlurDM_viz.png)
We provide qualitative comparisons of four baselines and their BlurDM-enhanced versions on the GoPro and HIDE test sets in this figure. The results show that BlurDM consistently produces sharper and more visually appealing deblurred results than "Baseline." The deblurred results of each BlurDM-enhanced backbone on each dataset can be download from [MIMO-UNet](<https://drive.google.com/file/d/1iP9hRE6SEUTzd8hormoGnUwWDCqNpoaq/view?usp=sharing>), [Stripformer](<https://drive.google.com/file/d/1f5qGy3iTv25sGT8s712LBecumMNIs3Yv/view?usp=sharing>), [FFTformer](<https://drive.google.com/file/d/1b7TWCo1Gr4NtBKxUOOQloFCXMiy2ptY7/view?usp=sharing>), [LoFormer](<https://drive.google.com/file/d/1iXg0SYYoZ1Z7GVbL-aHeeAIKisZO0jbD/view?usp=sharing>).
![Results](assets/BlurDM_result.png)
We compare the deblurring performance of four baselines and their BlurDM-enhanced versions, where “Baseline” and “BlurDM” refer to the deblurring performance without and with BlurDM, respectively. The results indicate that BlurDM consistently and significantly enhances deblurring performance.

## Overview
![Pipeline](assets/BlurDM_teaser.png)
We progressively add both noise and blur to a sharp image through a dual noise and blur diffusion process during forward diffusion. In the reverse process, BlurDM jointly denoises and deblurs the image, starting from Gaussian noise conditioned on the blurred input. 
![Overview](assets/BlurDM_overview.png)
We use BlurDM as a prior generation network to retain the diffusion model’s ability to learn high-quality, realistic image content while embedding the learned prior into the latent space of a deblurring network for effective and high-fidelity restoration.


## Citation
```
@article{he2026blurdm,
  title={Blurdm: A blur diffusion model for image deblurring},
  author={He, Jin-Ting and Tsai, Fu-Jen and Peng, Yan-Tsung and Chen, Min-Hung and Lin, Chia-Wen and Lin, Yen-Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={52196--52227},
  year={2026}
}
```
