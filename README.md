# Cancer Survival Prediction

## For Chulalongkorn AI Academy candidate screening

* Compatible with Python 3.6 and 3.7

## Problem introduction

* The ability to predict survival chance for cancer patient provides crucial information for making medical decision.
* Survival chance depends on patient and tumor characteristics.
* Patient basic characteristics were graded by clinicians through interviews and blood tests.
* Tumor characteristics were collected via imaging followed by manual annotations by doctors and automatic image processing.
* Can we predict the patient's survival chance with these information?

## Getting Started

Run the following command in your Terminal to install Project Dependencies

```sh
python install -r requirements.txt
```

To begin training the project, download and rename the dataset from the Kaggle page to the data/folder as so:

> data/radiomics_v1.csv # Clinical Data from 197 Patients  
> data/radiomics_v2.csv # 30 Patient with 3 Doctors Data  

Create a Cancer Survival Estimator using the following command:

```sh
python cancer.py
```

### Code Walkthrough

You can follow along the attached Jupyter Notebook titled `cancer_survival.ipynb` or compiled web-format `cancer_survival.html` 
on this project and see how data was treated before a Classification Model was trained on the data.
Rule #1 of Data Science is Garbage In, Garbage Out. This means that we need to identify how to work with the data given first.
The saved pickle file, `cancer_survival_estimator.pkl` can be used to predict chances of survival with great accuracy.

Sample Output:
  ```sh
  [INFO] Number of Wrong Predictions: 7 / 22
  [INFO] R2: -0.4667
  [INFO] Mean Squared Error: 0.3182
  [INFO] Root Mean Squared Error: 0.5641
  ```
