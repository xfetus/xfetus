# xfetus - A Python-based library for synthesising ultrasound images of fetal development (:baby: :brain: :robot:)
[![PyPI version](https://badge.fury.io/py/xfetus.svg)](https://badge.fury.io/py/xfetus)

xfetus is a Python-based library designed to synthesise fetal ultrasound images using state-of-the-art generative models, including GANs, transformers, diffusion models, and flow matching models. It also provides tools for assessing image synthesis quality through metrics such as FID, PSNR, SSIM, and Visual Turing Tests, along with access to relevant research publications.

## Installation
```
pip install xfetus
```

## Examples 
See [examples](examples) path with further instructions to run notebooks for data curation, classification, and models.  

## Development 
You can develop locally:
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (or [here](https://github.com/mxochicale/tools/blob/main/github/SSH.md))
* Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine:
```
cd && mkdir -p $HOME/repositories/xfetus && cd  $HOME/repositories/xfetus
git clone git@github.com:xfetus/xfetus.git
```

## Presentation
* [Good practices in AI/ML for Ultrasound Fetal Brain Imaging Synthesis](docs/event/README.md) for the deep learning and computer vision Journal Club on 1st of June 2023, 15:00 GMT.

## Articles 
### Citations
BibTeX to cite

```
@misc{iskandar-midl2023,
      author={
      	Michelle Iskandar and 
      	Harvey Mannering and 
      	Zhanxiang Sun and 
      	Jacqueline Matthew and 
      	Hamideh Kerdegari and 
      	Laura Peralta and 
      	Miguel Xochicale},
      title={Towards Realistic Ultrasound Fetal Brain Imaging Synthesis}, 
      year={2023},
      eprint={2304.03941},
      archivePrefix={arXiv},
      publisher = {arXiv},
      url = {https://arxiv.org/abs/2304.03941},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
``` 

```
@misc{bautista-miua2022,
  author = {Bautista, Thea and 
            Matthew, Jacqueline and 
            Kerdegari, Hamideh and 
            Peralta, Laura and 
            Xochicale, Miguel},
  title = {Empirical Study of Quality Image Assessment for Synthesis of Fetal Head Ultrasound Imaging with DCGANs},  
  year = {2022},
  eprint={2206.01731},
  archivePrefix={arXiv},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2206.01731},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

## Contributors
Thanks goes to all these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):  
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/sfmig"><img src="https://avatars1.githubusercontent.com/u/33267254?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Sofia Miñano</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/xfetus/xfetus/commits?author=sfmig" title="Code">💻</a> 
			<a href="https://github.com/xfetus/xfetus/commits?author=sfmig" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/seansunn"><img src="https://avatars1.githubusercontent.com/u/91659063?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Zhanxiang (Sean) Sun</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/xfetus/xfetus/commits?author=seansunn" title="Code">💻</a> 
			<a href="https://github.com/xfetus/xfetus/commits?author=seansunn" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/harveymannering"><img src="https://avatars1.githubusercontent.com/u/60523103?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Harvey Mannering</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/xfetus/xfetus/commits?author=harveymannering" title="Code">💻</a> 
			<a href="https://github.com/xfetus/xfetus/commits?author=harveymannering" title="Research">  🔬 🤔  </a>
	</td>
    <!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/michellepi"><img src="https://avatars1.githubusercontent.com/u/57605186?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Michelle Iskandar</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
            <a href="https://github.com/xfetus/xfetus/commits?author=michellepi" title="Code">💻</a>
			<a href="https://github.com/xfetus/xfetus/commits?author=michellepi" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/xfetus"><img src="https://avatars1.githubusercontent.com/u/11370681?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>Miguel Xochicale</b></sub>          
			<br />
		</a>
			<a href="https://github.com/xfetus/xfetus/commits?author=mxochicale" title="Code">💻</a> 
			<a href="ttps://github.com/budai4medtech/xfetus/commits?author=mxochicale" title="Documentation">📖  🔧 </a>
	</td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This work follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.  
Contributions of any kind welcome!
