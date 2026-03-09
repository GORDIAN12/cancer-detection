# Breast Cancer Data Analyzer

This project is a **data analysis application for breast cancer datasets**, focused on exploring tumor characteristics such as **texture** and **perimeter** to compare **benign** and **malignant** diagnoses.

The application is built using **Python** and **Streamlit**, providing interactive visualizations, statistical analysis, probability distributions, and confidence intervals to better understand tumor-related variables.

---

## Project Objective

The goal of this project is to analyze breast tumor characteristics using statistical methods and data visualization techniques in order to identify patterns and differences between **benign (B)** and **malignant (M)** cases.

---

## Features

- Load datasets from **CSV** and **Excel**
- Separate **benign** and **malignant** tumor cases
- Statistical analysis of tumor **texture**
- Histogram and density visualization
- Normal distribution fitting
- Probability Density Function (PDF) estimation
- Calculation of:
  - mean
  - standard deviation
  - standard error
  - confidence intervals
- Sampling simulation with different sample sizes
- Comparative visualization between benign and malignant tumors
- Exploratory **3D plots**
- Interactive interface built with **Streamlit**

---

## Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **SciPy**
- **OpenPyXL**

---

## Dataset Structure

The project works with files such as:

- `TEXTURA.csv`
- `PERIMETER.xlsx`
- `p.csv`
- `data.xlsx`

Example variables:

| Column | Description |
|------|-------------|
| diagnosis | Tumor diagnosis (B = benign, M = malignant) |
| texture | Texture measurement of the tumor |
| perimeter | Tumor perimeter measurement |
| id | Record identifier |
| num | Index value |

---

## System Workflow

1. Load data from CSV and Excel files.
2. Filter observations by diagnosis (benign or malignant).
3. Extract relevant variables such as **texture** and **perimeter**.
4. Generate histograms to visualize distributions.
5. Fit normal distributions to the data.
6. Calculate descriptive statistics.
7. Compute confidence intervals for the mean.
8. Compare benign and malignant distributions.
9. Simulate sampling distributions using different sample sizes.
10. Display results using an interactive Streamlit interface.

---

## Visualizations

### Texture Distribution
Histogram showing the distribution of tumor texture values with a fitted probability density function.

### Benign vs Malignant Comparison
Overlay histograms comparing texture distributions for benign and malignant tumors.

### Sampling Simulation
Multiple random samples are generated to observe how the **sampling mean distribution** behaves as the sample size increases.

### 3D Exploratory Plots
3D visualizations are created using tumor variables such as **perimeter** and **texture**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
