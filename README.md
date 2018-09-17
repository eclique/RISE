# RISE
This repository contains source code necessary to reproduce some of the main results in the paper:

[Vitali Petsiuk](http://cs-people.bu.edu/vpetsiuk/), [Abir Das](http://cs-people.bu.edu/dasabir/), [Kate Saenko](http://ai.bu.edu/ksaenko.html) (BMVC, 2018) <br>
[RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421)

**If you use this software in an academic article, please consider citing:**

    @inproceedings{Petsiuk2018rise,
      title = {RISE: Randomized Input Sampling for Explanation of Black-box Models},
      author = {Vitali Petsiuk and Abir Das and Kate Saenko},
      booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
      year = {2018}
    }

For more information regarding the paper, please visit http://cs-people.bu.edu/vpetsiuk/rise/.

## Method overview
To generate a saliency map for model's prediction, RISE queries black-box model on multiple randomly masked versions of input.
After all the queries are done we average all the masks with respect to their scores to produce the final saliency map. The idea behind this is that whenever a mask preserves important parts of the image it gets higher score, and consequently has a higher weight in the sum.
![](https://eclique.github.io/rep-imgs/RISE/rise-overview.png)

## Repository contents
* The whole idea is implemented in [Easy_start](Easy_start.ipynb) notebook, it's done in Keras but is really easy to modify for any framework, since the method itself is model-agnostic.
* [Saliency](Saliency.ipynb) notebook demonstrates the usage of RISE class optimized for PyTorch.
* [Evaluation](Evaluation.ipynb) notebook displays another contribution of the paper: *Causal metrics*.

## Examples
![](https://eclique.github.io/rep-imgs/RISE/example.png)
![](https://eclique.github.io/rep-imgs/RISE/goldish.gif)
