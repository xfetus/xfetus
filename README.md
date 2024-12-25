# xfetus -- :baby: :brain: :robot: -- A Python-based library for synthesising ultrasound images of fetal development 
[![PyPI version](https://badge.fury.io/py/xfetus.svg)](https://badge.fury.io/py/xfetus)

xfetus is a Python-based library designed to synthesise fetal ultrasound images using state-of-the-art generative models, including GANs, transformers, diffusion models, and flow matching models. It also provides tools for assessing image synthesis quality through metrics such as FID, PSNR, SSIM, and Visual Turing Tests, along with access to relevant research publications.

## :nut_and_bolt: Installation
### Install library
```
uv venv --python 3.12 # create uv VE
pip install xfetus
source .venv/bin/activate #activate VE
```

### Dev installation
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv pip install -e ".[test,learning]" # Install the package in editable mode
uv pip list --verbose #check versions
pre-commit run -a #pre-commit hooks
```
See further details [here](docs/dependencies).

## :school_satchel: Examples 
See [examples](docs/examples/) path with further instructions to run notebooks for data curation, classification, and models.  

## :scroll: Articles 
> Iskandar, Michelle, Harvey Mannering, Zhanxiang Sun, Jacqueline Matthew, Hamideh Kerdegari, Laura Peralta, and Miguel Xochicale. **"Towards realistic ultrasound fetal brain imaging synthesis."** arXiv preprint arXiv:2304.03941 (2023). Published in Medical Imaging with Deep Learning, MIDL 2023 Short paper track. Nashville, TN, US  Jul 10 2023.
[[Github-repository]](https://github.com/xfetus/midl2023); 
[[arXiv-preprint]](https://arxiv.org/abs/2304.03941); 
[[open-review]](https://openreview.net/forum?id=mad9Y_7khs); 
[[google-citations]](https://scholar.google.com/scholar?cites=12233870367431892152).

<details>

<summary>See bibtex to cite paper:</summary>

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
      copyright = 
	 	{Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
``` 

</details>


> Bautista, Thea, Jacqueline Matthew, Hamideh Kerdegari, Laura Peralta Pereira, and Miguel Xochicale. **"Empirical study of quality image assessment for synthesis of fetal head ultrasound imaging with dcgans."** arXiv preprint arXiv:2206.01731 (2022). Published in the 26th Conference on Medical Image Understanding and Analysis (MIUA 2022), Cambridge, 27-29 July 2022.
[[github-repository]](https://github.com/xfetus/miua2022); 
[[arXiv-preprint]](https://arxiv.org/abs/2206.01731); 
[[google-citations]](https://scholar.google.com/scholar?cites=3216210477950210889); 
[[YouTube-video-poster-presentation]](https://www.youtube.com/watch?v=wNKgScMzjPY).

<details>

<summary>See bibtex to cite paper:</summary>

```
@misc{bautista-miua2022,
  author = {Bautista, Thea and 
            Matthew, Jacqueline and 
            Kerdegari, Hamideh and 
            Peralta, Laura and 
            Xochicale, Miguel},
  title = {Empirical Study of Quality Image Assessment for 
  			Synthesis of Fetal Head Ultrasound Imaging with DCGANs},  
  year = {2022},
  eprint={2206.01731},
  archivePrefix={arXiv},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2206.01731},
  copyright = 
  	{Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
</details>


## :clapper: Presentations
* [Good practices in AI/ML for Ultrasound Fetal Brain Imaging Synthesis](docs/event/README.md) for the deep learning and computer vision Journal Club on 1st of June 2023, 15:00 GMT.

## :busts_in_silhouette: Contributors
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

## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (or [here](https://github.com/mxochicale/tools/blob/main/github/SSH.md))
* Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine:
```
cd && mkdir -p $HOME/repositories/xfetus && cd  $HOME/repositories/xfetus
git clone git@github.com:xfetus/xfetus.git
```
