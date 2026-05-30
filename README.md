# Machine-Learning

Graduate coursework in supervised and unsupervised learning, model evaluation, feature engineering, and deep learning for classification and regression, with business and computer-vision applications.

---

## 1. Title and Summary

**Machine Learning**  
Northwestern University M.S. in Data Science (Data Engineering specialization): end-to-end ML workflow from data prep and baseline models through tree ensembles, dimensionality reduction, clustering, and TensorFlow/Keras neural networks on tabular credit-risk data, steam benchmarks, iris, Fashion-MNIST, and satellite/cat-dog imagery.

---

## 2. Concepts and Methods

- **ML framework and tabular EDA:** load and inspect loan default datasets (`HMEQ_Loss.csv`); target variables for bad-flag classification and loss amount regression; missing-value and feature exploration (`Machine Learning.ipynb`)
- **Linear and logistic regression:** OLS on steam heating data with R² and MSE; logistic regression on steam logit data (`Linear_Logistic_Regression.ipynb`)
- **Feature scaling / transformation:** MinMaxScaler and StandardScaler on HMEQ features; compare normalized vs. standardized feature frames (`Transform.ipynb`, `Transform.py`)
- **Regression model selection:** stepwise and sequential feature selection with mlxtend `SequentialFeatureSelector`; compare linear, tree, forest, boosting, and Keras dense models; ROC/AUC curves for classifiers (`Regression_Models_Stepwise_Sequential_Feature_Selection.ipynb`)
- **Tree-based methods:** decision trees, random forests, gradient boosting for classification and regression; confusion matrices and classification reports (`DecisionTrees_RandomForests_GradientBoosting_Receiver_Operator_Characteristic_Curve.ipynb`)
- **Unsupervised learning:** K-means on iris and HMEQ with silhouette and Calinski-Harabasz scores; elbow-style cluster evaluation (`KMeans Clustering.ipynb`, `K_Means_Clustering.py`, `KMeans PCA.ipynb`, `K_Means_PCA.py`)
- **Dimensionality reduction:** PCA scree plots, variance explained, component loadings, 2D projection by species (`Principle Component Analysis.ipynb`, `Principal_Component_Analysis.py`)
- **Fully connected neural networks (tabular):** Keras Sequential dense/dropout models on HMEQ for binary default classification and loss regression; architecture and activation sweeps (`Deep Learning with Artificial Neural Networks.ipynb`, overlap in stepwise/ROC notebook)
- **Fashion-MNIST MLP/CNN baselines:** load `fashion_mnist`; flatten/normalize pixels; dense and convolutional experiments (`Neural_Networks.ipynb`, `Neural_Network_MNIST_Fashion_Data.ipynb`)
- **Computer vision pipeline:** OpenCV grayscale load/resize; cat/dog and satellite image folders; pickle feature tensors (`X.pickle`, `Y.pickle`); pixel normalization comparisons (`Data Normalization for Image Classification.ipynb`, `Satellite Imagery Machine Learning Image Recognition Pixel Normalization.ipynb`)
- **Convolutional neural networks:** Conv2D + MaxPooling stacks vs. dense baselines on pickled image tensors; train/validation split; softmax binary classification (`Deep Learning with Convolutional Neural Networks.ipynb`)
- **Model inference on new images:** load saved Keras model (`TFNN.pet.model`); preprocess and score held-out pet imagery (`Applying CNN to New Data.ipynb`)

**Course syllabus topic not represented in committed artifacts:** recurrent neural networks (no RNN/LSTM notebooks in this repository)

**Data dependencies:** coursework CSVs (`HMEQ_Loss.csv`, `Steam_Linear_Data.csv`, `Steam_Logit_Data.csv`, `IRIS.csv`), image directories, pickle tensors, and saved Keras models are referenced in notebooks but not bundled in the repository

**Out of scope for this repo:** production BigQuery ML serving (see **Analytics-Applications-Engineering**); inferential statistics in R (see **Statistics**); prescriptive LP/simulation (see **Decision-Analytics**); algorithm fundamentals without ML framing (see **Data-Engineering-Algorithms**); large-scale cloud feature pipelines (see **Systems-Engineering**, **Data-Miners**).

---

## 3. Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3 |
| Environment | Jupyter Notebook |
| Classical ML | scikit-learn (linear/logistic regression, trees, forests, gradient boosting, KMeans, PCA, metrics) |
| Feature selection | mlxtend (`SequentialFeatureSelector`) |
| Deep learning | TensorFlow 2 / Keras |
| Vision | OpenCV (`cv2`), pickle serialization |
| Data / viz | pandas, NumPy, matplotlib, seaborn |
| Scripts | `Transform.py`, `K_Means_Clustering.py`, `K_Means_PCA.py`, `Principal_Component_Analysis.py` |

---

## 4. Structure

```
Machine-Learning/
├── Machine Learning.ipynb
├── Linear_Logistic_Regression.ipynb
├── Transform.ipynb
├── Regression_Models_Stepwise_Sequential_Feature_Selection.ipynb
├── DecisionTrees_RandomForests_GradientBoosting_Receiver_Operator_Characteristic_Curve.ipynb
├── KMeans Clustering.ipynb
├── KMeans PCA.ipynb
├── Principle Component Analysis.ipynb
├── Deep Learning with Artificial Neural Networks.ipynb
├── Neural_Networks.ipynb
├── Neural_Network_MNIST_Fashion_Data.ipynb
├── Data Normalization for Image Classification.ipynb
├── Satellite Imagery Machine Learning Image Recognition Pixel Normalization.ipynb
├── Deep Learning with Convolutional Neural Networks.ipynb
├── Applying CNN to New Data.ipynb
├── Transform.py
├── K_Means_Clustering.py
├── K_Means_PCA.py
├── Principal_Component_Analysis.py
└── README.md
```

- **Organization:** notebooks grouped by method (tabular supervised → unsupervised → deep learning tabular → vision); four standalone `.py` scripts mirror notebook exercises
- **Reusable modules:** none packaged; helper functions (e.g., ROC plotting, image loaders) defined inline per assignment
- **Engineering practice:** train/test splits, ROC/AUC model comparison, sequential feature selection, cluster validation metrics, scree-driven PCA truncation, CV-based image preprocessing pipelines, saved-model inference on new files

---

**Course context:** Northwestern University, M.S. in Data Science, Data Engineering specialization (MSDS 422)  
**Repository:** https://github.com/EAName/Machine-Learning
