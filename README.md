# A Neural Algorithm of Artistic Style implementation - Neural Style Transfer
This is an implementation of the research paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576.pdf) written by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.

## Introduction
To quote authors Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, *"in light of the striking similarities between performance-optimised artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of how humans create and perceive artistic imagery*. 

The idea of *Neural Style Transfer* is taking a white noise as an input image, changing the input in such a way that it resembles the *content* of the content image and the *texture/artistic style* of the style image to reproduce it as a new artistic stylized image. 

We define two distances, one for the *content* that measures how different the content between the two images is, and one for *style* that measures how different the style between the two images is. The aim is to transform the white noise input such that the the content-distance and style-distance is minimized (with the content and style image respectively). 

## Usage Guidelines

- Cloning the Repository: 

        git clone https://github.com/srijarkoroy/ArtiStyle
        
- Entering the directory: 

        cd ArtiStyle
        
- Setting up the Python Environment with dependencies:

        pip install -r requirements.txt

- Running the file:

        python3 test.py
        
**Note**: Before running the test file please ensure that you mention a valid path to a content and style image and also set path='path to save the output image' if you want to save your image

Check out the demo notebook <a href = 'https://github.com/srijarkoroy/ArtiStyle/blob/main/demo/demo_nb.ipynb'>here</a>.

## Results from implementation

Content Image | Style Image | Output Image |
:-------------: | :---------: | :-----: |
<img src="results/input/content.jpg" height=200 width=200>| <img src="results/input/style.jpg" height=200 width=200>| <img src="results/output/result.jpeg" height=200 width=200> |
<img src="results/input/content2.jpeg" height=200 width=200>| <img src="results/input/style2.jpg" height=200 width=200>| <img src="results/output/result2.jpg" height=200 width=200> |
<img src="results/input/content3.jpg" height=200 width=200>| <img src="results/input/style3.jpg" height=200 width=200>| <img src="results/output/result3.jpg" height=200 width=200> |

<hr>

## Contributors

- <a href = "https://github.com/srijarkoroy">Srijarko Roy</a>
- <a href = "https://github.com/indiradutta">Indira Dutta</a>
