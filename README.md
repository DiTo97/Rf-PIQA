# Rf-PIQA

A machine learning toolkit for **R**eference-**f**ree **P**anorama **I**mage **Q**uality **A**ssessment (Rf-PIQA) modelling.

## 🚀 installation

```shell
python -m pip git+https://github.com/DiTo97/Rf-PIQA.git
```

## 🌟 overview

The toolkit supports two types of panorama image quality assessment (IQA):
- **reference (teacher) mode:** The model sees both a high-resolution panorama and its constituent low‐resolution images.
- **reference-less (student) mode:** The model only sees the high-resolution panorama.

A fully trained reference PIQA model shall be distilled into a reference‐less model via teacher-student training.

The toolkit supports two types of regression head:
- A simple "value estimate" head (fully connected layer).
- A PIVEN head for prediction intervals along with the value, and its corresponding loss[^1].

The PIVEN head is more expensive at training time, but enables confidence estimates on its predictions.

## 📄 documentation

The toolkit's documentation is hosted as a [GitHub wiki](https://github.com/DiTo97/Rf-PIQA/wiki).

## 🤝 contributing 

contributions to **Rf-PIQA** are welcome!

feel free to submit pull requests or open issues on our repository.

## 📄 license

see the [LICENSE](LICENSE) file for more details.

[^1]: [Simhayev et al., PIVEN: A Deep Neural Network for Prediction Intervals with Specific Value Prediction, 2020](https://arxiv.org/abs/2006.05139)
