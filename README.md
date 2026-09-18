# 📍 KNN Visualizer — Interactive K-Nearest Neighbors Demonstration

An interactive **Streamlit-based educational application** for demonstrating the working of the **K-Nearest Neighbors (KNN)** classification algorithm.

The application allows users to:

* Load a built-in dataset or upload their own CSV dataset
* Select the target column
* Select **1, 2, or 3 numerical features**
* Enable or disable feature standardization
* Enter a **single test data point**
* Select the value of **K**
* Predict the class of the test point
* Display the **K nearest training instances**
* Display the distance of each nearest neighbor
* Visualize the dataset and test point in **1D, 2D, or 3D**
* Visually connect the test point to its nearest neighbors
* Demonstrate how **feature scaling can affect KNN**

The project is particularly suitable for **Machine Learning courses, classroom demonstrations, laboratory sessions, tutorials, and self-learning**.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [What is KNN?](#-what-is-knn)
* [Why This Application?](#-why-this-application)
* [Key Features](#-key-features)
* [Application Workflow](#-application-workflow)
* [Supported Datasets](#-supported-datasets)
* [Dataset Upload](#-dataset-upload)
* [Feature Selection](#-feature-selection)
* [K Selection](#-k-selection)
* [Test Point](#-test-point)
* [Feature Scaling](#-feature-scaling)
* [Visualization](#-visualization)
* [Distance Calculation](#-distance-calculation)
* [Prediction](#-prediction)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Using the Application](#-using-the-application)
* [Example: Iris Dataset](#-example-iris-dataset)
* [Understanding the Output](#-understanding-the-output)
* [Understanding Feature Scaling](#-understanding-feature-scaling)
* [Custom CSV Dataset Requirements](#-custom-csv-dataset-requirements)
* [Important Notes](#-important-notes)
* [Educational Use Cases](#-educational-use-cases)
* [Possible Classroom Activities](#-possible-classroom-activities)
* [Limitations](#-limitations)
* [Possible Future Enhancements](#-possible-future-enhancements)
* [Contributing](#-contributing)
* [License](#-license)
* [Author](#-author)

---

# 🔎 Overview

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm commonly used for classification and regression.

Unlike many parametric models, KNN does not learn an explicit mathematical model during training. Instead, it stores the training observations and makes a prediction for a new data point based on the observations that are closest to it.

The central idea is:

> **Similar data points tend to have similar labels.**

For classification, KNN identifies the `K` closest training instances and uses **majority voting** to determine the predicted class.

For example, if:

```text
K = 5
```

and the five nearest neighbors have labels:

```text
Class A
Class A
Class B
Class A
Class B
```

then:

```text
Class A = 3
Class B = 2
```

Therefore, the predicted class is:

```text
Class A
```

This application makes this process visually observable.

---

# 🤖 What is KNN?

KNN stands for:

**K-Nearest Neighbors**

It is a **supervised learning algorithm**.

For a new test point:

1. Calculate its distance from every training point.
2. Sort the training points according to distance.
3. Select the closest `K` points.
4. Examine their class labels.
5. Use majority voting.
6. Assign the majority class to the test point.

---

## Example

Suppose we have the following two-dimensional dataset:

| Point | Feature 1 | Feature 2 | Class |
| ----- | --------: | --------: | ----- |
| P1    |         2 |         3 | A     |
| P2    |         3 |         4 | A     |
| P3    |         8 |         7 | B     |
| P4    |         9 |         8 | B     |
| P5    |         4 |         3 | A     |

Suppose the test point is:

```text
X = (3, 3)
```

For:

```text
K = 3
```

KNN finds the three closest points.

If their classes are:

```text
A
A
A
```

the prediction becomes:

```text
A
```

The application allows this process to be seen directly on a plot.

---

# 🎯 Why This Application?

KNN is relatively easy to understand mathematically, but students often find it difficult to visualize:

* What "nearest" actually means
* How changing `K` changes the prediction
* Why distance matters
* Why feature scaling matters
* Which observations are actually selected as neighbors
* How the test point relates to the training data

This application addresses these issues by providing an interactive visualization.

Instead of only showing:

```text
Prediction = Class A
```

the application shows:

```text
Test Point
     ↓
Calculate distances
     ↓
Find nearest observations
     ↓
Select K neighbors
     ↓
Majority voting
     ↓
Prediction
```

The selected neighbors are highlighted directly in the visualization.

---

# ✨ Key Features

## 1. Built-in datasets

The application includes:

* Iris
* Wine
* Breast Cancer

These datasets are provided through `scikit-learn`.

---

## 2. CSV dataset upload

Users can upload their own `.csv` dataset.

The application automatically reads the CSV file using Pandas.

---

## 3. Target column selection

The user can select which column represents the class/target variable.

Example:

```text
sepal length
sepal width
petal length
petal width
species
```

The user can select:

```text
species
```

as the target column.

---

## 4. Feature selection

The user can select between:

```text
1 feature
2 features
3 features
```

Only numerical features are offered for selection because KNN distance calculations require numerical representations.

---

## 5. K selection

The user can interactively select the value of `K`.

For example:

```text
K = 1
K = 3
K = 5
K = 7
...
```

This makes it easy to observe how changing `K` affects the selected neighbors and prediction.

---

## 6. Single test point

The application intentionally accepts **one test point at a time**.

This makes the demonstration easier to understand.

The user enters one value for every selected feature.

For example:

```text
Feature 1 = 5.2
Feature 2 = 3.1
Feature 3 = 1.5
```

This represents one test instance.

---

## 7. Feature scaling

The application includes a checkbox:

```text
☑ Standardize features
```

When enabled, `StandardScaler` is applied before calculating distances.

This is particularly important for demonstrating one of the most important practical issues with KNN:

> **KNN is sensitive to feature scale.**

---

## 8. 1D visualization

When one feature is selected, the application provides a one-dimensional representation of the observations.

The test point and its nearest neighbors are highlighted.

---

## 9. 2D visualization

When two features are selected, the application displays a two-dimensional scatter plot.

The plot shows:

* Training observations
* Classes
* K nearest neighbors
* Test point
* Connections between the test point and nearest neighbors

---

## 10. 3D visualization

When three features are selected, the application automatically creates an interactive 3D scatter plot.

The user can rotate and zoom the visualization.

---

## 11. Neighbor table

The application displays the selected nearest neighbors in a table.

The table contains:

* Neighbor rank
* Distance
* Feature values
* Target/class

Example:

| Rank | Distance | Feature 1 | Feature 2 | Class  |
| ---: | -------: | --------: | --------: | ------ |
|    1 |     0.31 |       5.1 |       3.2 | Setosa |
|    2 |     0.42 |       5.0 |       3.4 | Setosa |
|    3 |     0.57 |       5.2 |       3.1 | Setosa |

---

# 🔄 Application Workflow

The application follows this workflow:

```text
              ┌───────────────────┐
              │    Load Dataset   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Select Target     │
              │ Column            │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Select 1–3        │
              │ Features          │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Scaling ON/OFF    │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Select K           │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Enter Test Point  │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Calculate         │
              │ Distances         │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Select K Nearest  │
              │ Neighbors         │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Majority Voting   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Prediction        │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Visualization     │
              └───────────────────┘
```

---

# 📊 Supported Datasets

The application currently supports three built-in datasets.

## Iris

The Iris dataset is a classic classification dataset containing measurements of iris flowers.

Typical features include:

```text
sepal length
sepal width
petal length
petal width
```

The target represents the iris species.

It is particularly useful for classroom demonstrations because it has a small number of features and observations.

---

## Wine

The Wine dataset contains chemical measurements associated with different wine classes.

It provides a larger number of numerical features and is useful for demonstrating:

* Feature selection
* High-dimensional data
* Feature scaling
* The effect of choosing different feature combinations

---

## Breast Cancer

The Breast Cancer dataset contains numerical measurements used for binary classification.

It can be useful for demonstrating KNN with:

* Multiple numerical features
* Different feature scales
* Standardization

---

# 📁 Dataset Upload

The application also supports custom CSV files.

Select:

```text
Upload CSV
```

from the sidebar.

Then upload a `.csv` file.

Example:

```csv
Age,Income,SpendingScore,Class
25,30000,70,A
35,50000,40,B
28,35000,65,A
45,80000,30,B
```

The application will allow the user to select:

```text
Target column → Class
```

and numerical features such as:

```text
Age
Income
SpendingScore
```

---

# 🎛️ Feature Selection

The application allows a maximum of three features.

### One feature

```text
X = [Feature 1]
```

Visualization:

```text
1D
```

### Two features

```text
X = [Feature 1, Feature 2]
```

Visualization:

```text
2D
```

### Three features

```text
X = [Feature 1, Feature 2, Feature 3]
```

Visualization:

```text
3D
```

This design keeps the visualization understandable while still demonstrating the multidimensional nature of KNN.

---

# 🔢 K Selection

The parameter `K` determines how many neighbors participate in the prediction.

For example:

```text
K = 1
```

means only the closest training instance is considered.

```text
K = 5
```

means the five closest training instances are considered.

```text
K = 10
```

means the ten closest training instances are considered.

Changing `K` can change:

* The selected neighbors
* The majority vote
* The final prediction

This makes `K` an excellent parameter for classroom experimentation.

---

# 🧪 Test Point

The application accepts only one test point.

Suppose two features are selected:

```text
Feature 1 = 5.2
Feature 2 = 3.1
```

The application creates:

```text
X_test = [5.2, 3.1]
```

The test point is then compared with all training observations.

For three selected features:

```text
X_test = [5.2, 3.1, 1.4]
```

---

# 📏 Distance Calculation

The default KNN distance used by `scikit-learn` is Euclidean distance.

For two points:

```text
A = (x₁, x₂)
B = (y₁, y₂)
```

the Euclidean distance is:

$$
d(A,B)=
\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}
$$

For three features:

$$
d(A,B)=
\sqrt{
(x_1-y_1)^2+
(x_2-y_2)^2+
(x_3-y_3)^2
}
$$

The same principle extends to more dimensions.

The application calculates the neighbors using:

```python
knn.kneighbors()
```

---

# ⚖️ Feature Scaling

Feature scaling is one of the most important concepts demonstrated by this application.

Consider two features:

```text
Age:       18 – 70
Income:    20,000 – 200,000
```

The income feature has a much larger numerical range.

Because KNN relies on distance, the larger-scale feature can have a much greater effect on the distance calculation.

For example:

$$
d =
\sqrt{
(Age_1-Age_2)^2+
(Income_1-Income_2)^2
}
$$

The income difference can dominate the distance.

---

## Standardization

When scaling is enabled, the application uses:

```python
StandardScaler()
```

The standardization formula is:

$$
z =
\frac{x-\mu}{\sigma}
$$

where:

* \(x\) = original value
* \(\mu\) = mean
* \(\sigma\) = standard deviation

After standardization, the features are represented on a more comparable scale.

---

# 🔬 Why Compare Scaling ON and OFF?

One of the intended educational uses of this application is to perform the following experiment.

### Experiment

1. Select a dataset.
2. Select two features.
3. Select `K = 5`.
4. Enter a test point.
5. Turn scaling **OFF**.
6. Record the nearest neighbors.
7. Turn scaling **ON**.
8. Compare the nearest neighbors.

You may observe that:

```text
Scaling OFF
     ↓
Neighbor set A
     ↓
Prediction A
```

while:

```text
Scaling ON
     ↓
Neighbor set B
     ↓
Prediction B
```

The result may or may not change depending on the dataset and test point.

The important concept is that **distance relationships can change after feature transformation**.

---

# 📈 Visualization

The application automatically selects the visualization based on the number of selected features.

## 1 Feature

A one-dimensional representation is displayed.

The test point is shown separately from the training observations.

---

## 2 Features

A 2D scatter plot is generated.

Conceptually:

```text
Feature 2
   ↑
   |
   |       ○
   |   ○       ○
   |       ★
   |    ○  ─── ○
   |
   +----------------→ Feature 1
```

Where:

```text
★ = Test point
○ = Training observations
```

The nearest neighbors are highlighted and connected to the test point.

---

## 3 Features

A 3D interactive scatter plot is generated.

The three selected features become:

```text
X-axis → Feature 1
Y-axis → Feature 2
Z-axis → Feature 3
```

The plot can be:

* Rotated
* Zoomed
* Explored interactively

---

# 🧮 Prediction

For classification, KNN uses majority voting.

Suppose:

```text
K = 5
```

and the nearest neighbors are:

| Neighbor | Class |
| -------- | ----- |
| 1        | A     |
| 2        | A     |
| 3        | B     |
| 4        | A     |
| 5        | B     |

The votes are:

```text
A → 3
B → 2
```

Therefore:

```text
Prediction = A
```

The application uses:

```python
KNeighborsClassifier(
    n_neighbors=k
)
```

for the prediction.

---

# 📋 Neighbor Information

The application also displays the actual nearest observations.

For example:

```text
Top 5 Nearest Neighbors
```

might produce:

| Rank | Distance | Feature 1 | Feature 2 | Target  |
| ---: | -------: | --------: | --------: | ------- |
|    1 |    0.214 |       5.1 |       3.2 | Class A |
|    2 |    0.327 |       5.0 |       3.1 | Class A |
|    3 |    0.451 |       5.3 |       3.3 | Class A |
|    4 |    0.592 |       6.0 |       3.0 | Class B |
|    5 |    0.634 |       5.9 |       3.1 | Class B |

This table provides the numerical explanation behind the visualization.

---

# 📂 Project Structure

A minimal project structure is:

```text
knn-visualizer/
│
├── app.py
├── README.md
└── requirements.txt
```

A recommended GitHub structure is:

```text
knn-visualizer/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── screenshots/
    ├── iris-2d.png
    ├── iris-3d.png
    └── scaling-comparison.png
```

---

# 🛠️ Technology Stack

The project uses the following technologies.

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Core programming language          |
| Streamlit    | Interactive web application        |
| Pandas       | Dataset handling                   |
| NumPy        | Numerical computation              |
| Scikit-learn | KNN, datasets, and scaling         |
| Plotly       | Interactive 1D/2D/3D visualization |

---

# 📦 Installation

## Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/knn-visualizer.git
```

Move into the project directory:

```bash
cd knn-visualizer
```

---

## Step 2 — Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install the dependencies directly:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

# 📄 requirements.txt

Create a file named:

```text
requirements.txt
```

with:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

For reproducible deployments, package versions can also be pinned.

---

# ▶️ Running the Application

Run:

```bash
streamlit run app.py
```

Streamlit will start a local web server.

The application can then be opened in a browser.

Typically:

```text
http://localhost:8501
```

---

# 🚀 Using the Application

## Step 1 — Select a dataset

From the sidebar:

```text
Dataset
    ↓
Built-in dataset
```

Choose:

```text
Iris
Wine
Breast Cancer
```

Alternatively select:

```text
Upload CSV
```

to load your own dataset.

---

## Step 2 — Select target

Choose the target/class column.

For Iris, this could be:

```text
Target
```

---

## Step 3 — Select features

Choose one, two, or three numerical features.

Example:

```text
sepal length
sepal width
petal length
```

---

## Step 4 — Choose scaling

Select:

```text
Standardize features
```

to enable standardization.

Leave it unchecked to use the original feature values.

---

## Step 5 — Select K

Choose the number of nearest neighbors.

Example:

```text
K = 5
```

---

## Step 6 — Enter the test point

Enter one value for each selected feature.

For example:

```text
sepal length = 5.1
sepal width  = 3.2
```

---

## Step 7 — Observe the result

The application displays:

```text
Prediction
K
Number of features
```

followed by:

```text
Top K Nearest Neighbors
```

and the visualization.

---

# 🌸 Example: Iris Dataset

A simple classroom demonstration can use the Iris dataset.

### Select:

```text
Dataset:
Iris
```

Target:

```text
Target
```

Features:

```text
Petal Length
Petal Width
```

K:

```text
5
```

Then enter a test point.

For example:

```text
Petal Length = 1.5
Petal Width  = 0.3
```

The application will:

1. Calculate the distance between the test point and training observations.
2. Identify the five closest observations.
3. Display their distances.
4. Display their class labels.
5. Perform majority voting.
6. Display the predicted class.
7. Highlight the neighbors in the plot.

---

# 🎓 Understanding the Output

The application provides three major outputs.

## 1. Prediction

Example:

```text
Prediction: setosa
```

This is the class predicted by KNN.

---

## 2. Neighbor Table

The table explains **which observations caused the prediction**.

This is useful for understanding KNN because the algorithm is instance-based.

---

## 3. Visualization

The visualization provides a geometric interpretation of the prediction.

Students can directly see:

```text
Test Point
     ↓
Nearby observations
     ↓
K nearest observations
     ↓
Majority class
```

---

# 🧑‍🏫 Educational Use Cases

This application can be used for teaching several Machine Learning concepts.

### KNN fundamentals

Demonstrate:

* Instance-based learning
* Lazy learning
* Distance-based classification
* Majority voting
* Choice of K

---

### Distance metrics

Demonstrate Euclidean distance:

$$
d =
\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

---

### Feature scaling

Demonstrate why preprocessing matters for distance-based algorithms.

---

### Curse of dimensionality

The application is intentionally limited to three visualization dimensions, but students can discuss what happens when KNN operates in much higher-dimensional spaces.

---

### Hyperparameter selection

Students can experiment with:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

and observe how the neighborhood changes.

---

# 🧪 Suggested Classroom Experiments

## Experiment 1 — Effect of K

Use the same test point.

Run:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

Record:

|  K | Prediction | Nearest Neighbor Pattern |
| -: | ---------- | ------------------------ |
|  1 |            |                          |
|  3 |            |                          |
|  5 |            |                          |
|  7 |            |                          |
|  9 |            |                          |

Discuss:

> Why can increasing K change the prediction?

---

# 🧪 Experiment 2 — Effect of Feature Scaling

Select two features with substantially different ranges.

Run the application twice:

```text
Scaling = OFF
```

and:

```text
Scaling = ON
```

Compare:

* Distances
* Nearest neighbors
* Prediction

Discuss:

> Why does changing the scale of a feature affect KNN?

---

# 🧪 Experiment 3 — Effect of Feature Selection

Keep `K` fixed.

Try different feature combinations.

For example:

```text
Feature 1 + Feature 2
```

then:

```text
Feature 1 + Feature 3
```

then:

```text
Feature 2 + Feature 3
```

Observe whether the nearest neighbors change.

---

# 🧪 Experiment 4 — Outliers

Use a dataset containing unusual observations.

Ask students:

> What happens when an outlier becomes one of the nearest neighbors?

---

# 🧪 Experiment 5 — Decision Boundaries

Use a suitable two-feature dataset and gradually change the test point.

Ask students to observe how the prediction changes as the test point moves through the feature space.

---

# ⚠️ Important Notes

## Numeric features

The current application is designed around numerical features.

Categorical features should be encoded before using them with Euclidean-distance-based KNN.

For example:

```text
Gender = Male/Female
```

should not simply be converted arbitrarily to:

```text
Male = 1
Female = 2
```

without considering whether that representation is appropriate.

---

## Missing values

Rows containing missing values in the selected features or target are removed before training.

For production applications, more sophisticated missing-value handling may be preferable.

---

## Target column

The target column should contain class labels suitable for classification.

Examples:

```text
0 / 1
A / B
cat / dog
setosa / versicolor / virginica
```

---

# ⚙️ Implementation Details

The application uses:

```python
KNeighborsClassifier
```

from:

```python
sklearn.neighbors
```

The model is trained using:

```python
knn.fit(X_model, y)
```

Prediction is performed using:

```python
knn.predict(test_model)
```

Nearest neighbors are obtained using:

```python
knn.kneighbors(test_model)
```

This returns:

```text
distances
indices
```

The indices are then used to identify the corresponding rows in the original dataset.

---

# 🧠 Why `kneighbors()` is Important

The prediction alone does not explain which observations influenced the decision.

For teaching purposes, `kneighbors()` is particularly useful because it provides:

```text
Distance
+
Index of neighbor
```

This allows the application to explicitly display the nearest observations.

Therefore, the application does not behave like a simple:

```text
Input → Prediction
```

black-box demo.

Instead, it exposes the neighborhood responsible for the prediction.

---

# 📐 Mathematical Interpretation

Given a test point:

$$
x =
(x_1,x_2,\ldots,x_n)
$$

and training point:

$$
x^{(i)} =
(x_1^{(i)},x_2^{(i)},\ldots,x_n^{(i)})
$$

the Euclidean distance is:

$$
d(x,x^{(i)})
=
\sqrt{
\sum_{j=1}^{n}
(x_j-x_j^{(i)})^2
}
$$

The training observations are ordered by distance.

The first `K` observations are selected.

For classification, the predicted class is:

$$
\hat{y}
=
\operatorname{mode}
\left(
y^{(1)},y^{(2)},\ldots,y^{(K)}
\right)
$$

where the \(y\)'s correspond to the selected nearest neighbors.

---

# 📚 Concepts Demonstrated

The application brings together the following concepts:

```text
Supervised Learning
        ↓
Classification
        ↓
KNN
        ↓
Distance Calculation
        ↓
Euclidean Distance
        ↓
Nearest Neighbors
        ↓
Majority Voting
        ↓
Prediction
        ↓
Feature Scaling
        ↓
Visualization
```

---

# 🔮 Possible Future Enhancements

The application can be extended with several features.

## 1. Regression mode

Add support for:

```text
KNN Classification
KNN Regression
```

For regression, instead of majority voting, the predictions of neighboring points can be averaged.

---

## 2. Distance metric selection

Allow users to select:

```text
Euclidean
Manhattan
Minkowski
```

and compare the resulting neighbors.

---

## 3. Distance weighting

Add:

```text
Uniform weights
Distance-based weights
```

using the `weights` parameter of `KNeighborsClassifier`.

---

## 4. Train-test split

Allow users to specify:

```text
Training %
Testing %
```

and evaluate KNN on a complete test set.

---

## 5. Accuracy evaluation

Display:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

---

## 6. Decision boundary visualization

For two-dimensional datasets, display the KNN decision regions.

This would provide a useful demonstration of how KNN partitions feature space.

---

## 7. Manual distance calculation

An educational mode could display:

```text
Distance to Point 1 = ...
Distance to Point 2 = ...
Distance to Point 3 = ...
...
```

and show the complete sorting process.

This would be particularly useful for teaching students how KNN works without relying on the implementation inside scikit-learn.

---

## 8. Manual KNN mode

A future version could provide two modes:

```text
Scikit-learn KNN
```

and:

```text
From-scratch KNN
```

This would allow students to compare the implementation with the underlying algorithm.

---

# 🚧 Limitations

This project is primarily designed for **educational visualization**, not production machine learning.

Current limitations include:

* Maximum of three visualized features
* Numerical features are required
* Classification is currently supported
* Missing rows are dropped rather than imputed
* No automated hyperparameter optimization
* No train/test evaluation interface
* No decision-boundary visualization
* No categorical feature encoding
* No cross-validation
* No model persistence

These limitations are intentional in the current version to keep the application focused on understanding the fundamental KNN algorithm.

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the application:

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/new-feature
```

3. Make your changes.
4. Commit the changes.

```bash
git add .
git commit -m "Add new feature"
```

5. Push the branch.

```bash
git push origin feature/new-feature
```

6. Open a Pull Request.

---

# 📝 Suggested Contribution Areas

Some useful contribution ideas include:

* KNN regression
* Additional datasets
* Additional distance metrics
* Decision boundaries
* Confusion matrix
* Cross-validation
* Manual KNN implementation
* Improved 3D visualization
* Better custom dataset validation
* Downloadable prediction results
* Interactive distance calculation
* Animation of the KNN process

---

# 📜 License

This project can be released under the **MIT License**.

If using the MIT License, add a file named:

```text
LICENSE
```

containing the appropriate MIT License text.

---

# 👨‍💻 Author

**Sawan Rai**

Assistant Professor, Computer Science and Engineering

Research interests include:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Information Retrieval
* Software Engineering

---

# ⭐ If You Find This Useful

If this project helps you understand or teach KNN, consider:

* ⭐ Starring the repository
* 🍴 Forking the repository
* 🐛 Reporting issues
* 💡 Suggesting improvements
* 🤝 Contributing enhancements

---

# 📌 Quick Start

For users who just want to run the application:

```bash
git clone https://github.com/YOUR_USERNAME/knn-visualizer.git

cd knn-visualizer

pip install -r requirements.txt

streamlit run app.py
```

Then open the Streamlit application in your browser.

---

## 🎓 Core Learning Objective

The primary objective of this project is to make the following KNN concept visually intuitive:

$$
\boxed{
\text{A test point is classified based on the classes of its K nearest training points.}
}
$$

By allowing the user to **change the features, test point, K, and feature scaling interactively**, the application turns KNN from a static algorithm into an experiment that students can explore.
