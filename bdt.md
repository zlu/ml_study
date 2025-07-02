# Bayesian Decision Theory: A Comprehensive Guide to Probabilistic Classification

Bayesian Decision Theory is a cornerstone of statistical pattern recognition, providing a robust framework for classifying objects based on probabilistic reasoning. It leverages Bayes' Theorem to make optimal decisions under uncertainty, making it a powerful tool in machine learning applications like image recognition, spam filtering, and medical diagnosis. This blog post explores Bayesian Decision Theory as presented in the provided lecture notes, delving into its mathematical foundations, practical implementations, and advanced extensions like Gaussian Mixture Models (GMM) and Naïve Bayes classifiers. We'll also include Python code samples to illustrate key concepts, ensuring a clear understanding for both beginners and advanced practitioners.

## Introduction to Bayesian Decision Theory

Bayesian Decision Theory addresses pattern classification by modeling the probability of a sample belonging to a specific class based on observed features. The lecture notes use a fish classification example, distinguishing between sea bass ($\omega_1$) and salmon ($\omega_2$), to illustrate the approach. Each class has a prior probability, $ p(\omega_1) $ and $ p(\omega_2) $, representing the likelihood of encountering each fish type without additional information. If $ p(\omega_1) = p(\omega_2) = 0.5 $, both classes are equally likely. In practice, prior probabilities are often estimated from data or domain knowledge.

When additional information, such as a feature like lightness measurement ($ x $), is available, Bayesian Decision Theory incorporates this evidence to compute the posterior probability, $ p(\omega_j | x) $, which updates the likelihood of each class given the feature. The decision rule then assigns the sample to the class with the highest posterior probability.

## Bayes' Theorem in Classification

Bayes' Theorem is the backbone of this approach, expressed as:

$$ p(\omega_j | x) = \frac{p(x | \omega_j) p(\omega_j)}{p(x)} $$

where:
- $ p(\omega_j | x) $: Posterior probability of class $\omega_j$ given feature $ x $.
- $ p(x | \omega_j) $: Class-conditional probability density, describing the distribution of feature $ x $ for class $\omega_j$.
- $ p(\omega_j) $: Prior probability of class $\omega_j$.
- $ p(x) $: Evidence, computed as $ p(x) = \sum_{j=1}^c p(x | \omega_j) p(\omega_j) $, where $ c $ is the number of classes.

For a two-class problem, the decision rule is:

- Choose $\omega_1$ if $ p(\omega_1 | x) > p(\omega_2 | x) $.
- Choose $\omega_2$ otherwise.

Since $ p(x) $ is a common scaling factor, the decision can be simplified to comparing $ p(x | \omega_1) p(\omega_1) $ and $ p(x | \omega_2) p(\omega_2) $.

### Example: Fish Classification

Suppose we have prior probabilities $ p(\omega_1) = 2/3 $ (sea bass) and $ p(\omega_2) = 1/3 $ (salmon). Given a lightness measurement $ x $, we compute the posterior probabilities using the class-conditional densities functions $ p(x | \omega_1) $ and $ p(x | \omega_2) $, which describe the lightness distribution for each fish type.

## Generalizing to Multiple Features and Classes

For real-world applications, we often deal with multiple features (forming a feature vector $\mathbf{x}$) and multiple classes ($\omega_1, \omega_2, \ldots, \omega_c$). The posterior probability becomes:

$$ p(\omega_j | \mathbf{x}) = \frac{p(\mathbf{x} | \omega_j) p(\omega_j)}{p(\mathbf{x})} $$

where $ p(\mathbf{x}) = \sum_{j=1}^c p(\mathbf{x} | \omega_j) p(\omega_j) $. The classifier assigns $\mathbf{x}$ to the class $\omega_i$ that maximizes the discriminant function:

$$ g_i(\mathbf{x}) = p(\omega_i | \mathbf{x}) $$

Alternative discriminant functions include:

$$ g_i(\mathbf{x}) = p(\mathbf{x} | \omega_i) p(\omega_i) $$
$$ g_i(\mathbf{x}) = \log p(\mathbf{x} | \omega_i) + \log p(\omega_i) $$

These are equivalent for classification, as they preserve the order of probabilities.

## Gaussian Density Functions

The lecture notes emphasize the multivariate normal (Gaussian) distribution for modeling class-conditional densities functions due to its mathematical tractability and prevalence in real-world data. For a univariate case:

$$ p(x | \omega) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) $$

For a $ d $-dimensional feature vector $\mathbf{x}$:

$$ p(\mathbf{x} | \omega) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) $$

where $\boldsymbol{\mu}$ is the mean vector, and $\Sigma$ is the covariance matrix.

### Parameter Estimation

In practice, $\boldsymbol{\mu}$ and $\Sigma$ are unknown and estimated from training data using maximum-likelihood estimation (MLE). For a dataset $ D $ with $ n $ samples $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$:

$$ \hat{\boldsymbol{\mu}} = \frac{1}{n} \sum_{k=1}^n \mathbf{x}_k $$
$$ \hat{\Sigma} = \frac{1}{n} \sum_{k=1}^n (\mathbf{x}_k - \hat{\boldsymbol{\mu}})(\mathbf{x}_k - \hat{\boldsymbol{\mu}})^T $$

The log-likelihood function is:

$$ l(\boldsymbol{\theta}) = \sum_{k=1}^n \ln p(\mathbf{x}_k | \boldsymbol{\theta}) $$

where $\boldsymbol{\theta} = (\boldsymbol{\mu}, \Sigma)$. Maximizing this yields the MLE estimates.

### Code Example: Gaussian Bayes Classifier

Here's a Python implementation using scikit-learn to classify data with a Gaussian assumption:

<xaiArtifact artifact_id="ff0954ba-e90f-4f91-9494-e391c34c9924" artifact_version_id="38edb466-27cd-47ef-9728-32b4b302d12c" title="gaussian_bayes_classifier.py" contentType="text/python">
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Simulated data: 200 samples, 2 features, 2 classes
np.random.seed(42)
class1 = np.random.multivariate_normal([-0.1055, -0.0974], [[1.0253, -0.0036], [-0.0036, 0.8880]], 100)
class2 = np.random.multivariate_normal([2.0638, 3.0451], [[1.1884, -0.013], [-0.013, 1.0198]], 100)
X = np.vstack((class1, class2))
y = np.array([0] * 100 + [1] * 100)

# Train Gaussian Naive Bayes classifier
gnb = GaussianNB(priors=[1/3, 2/3])
gnb.fit(X, y)

# Predict a new sample
new_sample = np.array([[0.5, 0.5]])
prediction = gnb.predict(new_sample)
print(f"Predicted class: {prediction[0]}")