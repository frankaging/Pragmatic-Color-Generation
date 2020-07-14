# Pragmatically Informative Color Generation with Contextual Modifiers

Grounding language in contextual information is crucial for fine-grained natural language understanding. One important task that involves grounding contextual modifiers is color generation. Given a reference color "green", and a modifier "bluey", how does one generate a color that could represent "bluey green"? We propose a computational pragmatics model that formulates this color generation task as a recursive game between speakers and listeners. In our model, a pragmatic speaker reasons about the inferences that a listener would make, and thus generates a modified color that is maximally informative to help the listener recover the original referents. Although such techniques have been successfully applied in cognitive science and computational linguistics reference-game tasks (e.g. learning of grounded contextual dependence in reference games), such pragmatic modeling has not been applied to tasks such as color generation, where one has to ground contextual modifiers to generate a sample from a large continuous space. In this paper, we show that incorporating pragmatic information into deep learning models provides significant improvements in performance compared with other state-of-the-art deep learning models. Our results also demonstrate an extensible pragmatic pipeline for other contextual image generation tasks.

# Task Definition
<img src="https://i.ibb.co/HgDg8KT/example.png" width="300">
  - You are given triples include (Reference Color Lable, Modifier, Target Color Label)
  - For color lables, you are given RGB vectors related to these colors. Each color will have a set of RGB vectors. It is not a **one to one** mapping.

# Deep Learning Models

> The deep learning models are developed based on previous works,
> and serve as baselines in our paper.

# Pragmatic Models
<img src="https://i.ibb.co/8bfgZ7F/pragmatic-model.png" width="500">

# Performance Results
<img src="https://i.ibb.co/y54h9MC/perf.png" width="500">

# Examples of the Color Generation Task
<img src="https://i.ibb.co/25bFGJk/result-detail.png" width="500">

### Installation

Our code requires python v3.6+ to run. You will also need various libraries such as pytorch, sklearn, etc..

Install the dependencies and devDependencies and start the server.

```sh
$ cd Pragmatic-Color-Generation
$ pip install requirements.txt
```

### How to run code?
With our pretrained models and preprocess datasets, you can go to our code directly, and:
```python
$ cd model
$ jupyter notebook
```
You can use the files to retrain models as well. Details about flags and settings are in the file.

To re-preprocess the dataset, you can use
```python
$ cd model
$ python dataset.py
```

License
----

MIT
