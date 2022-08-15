### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 97c8fe8a-21c0-11ec-1100-638723043ca3
begin
	using Distributions, Random, LinearAlgebra, Plots, StatsPlots
	using StatsFuns, FiniteDifferences, ForwardDiff, GLM, DataFrames
	plotly()
end

# ╔═╡ b1d6136d-403d-47b7-aa9d-60b59d0b15b6
md"""
### Question 2. Conditional expectation

Conditional expectation is usually written as $E[Y|X=x]$, note that capital letters are random variables while smaller case is a concrete value; *A good introduction to conditional expectation is Introduction to probability models by Sheldon Ross. You should read 2.4 and 3.1, 3.2, 3.3, and 3.4 if you do not know conditional expectation*

1. what is the definition of conditional expectation ?
2. is $E[Y|X=x]$ a function of $x$ or $y$ or $X$, $Y$ ?
3. prove $E\left[E[Y|X=x]\right] = E[Y]$, with respect to what random variable's distribution the outer expectation is calculated ?
"""

# ╔═╡ 7da12ea5-d615-4de1-a728-32c2e5980432
md"""
### Solution

Marginal expected value of $Y$ is:

$$E[Y] = \int y \cdot p(Y=y)dy$$

The discrete case is 

$$E[Y] = \sum_{y} y \cdot p(Y=y)$$

"""

# ╔═╡ 14f6f12b-096e-450b-bc42-4bce785e686c
md"""
Conditional expectation is:

$$E[Y|X=x] =\int y \cdot p(Y=y|X=x) dy$$


$$p(Y=y|X=x) = \frac{p(X=x, Y=y)}{p(X=x)}$$


$$p(Y=y) = \sum_x p(X=x, Y=y)$$
"""

# ╔═╡ e67efa89-9b1c-44f8-a933-117592524609
md"""
Because $E[Y|X=x]=f(x)$, i.e. a function of $x$; what is the expectation of $f(x)$?

$$E[f(x)] = \int f(x) p(x) dx$$

$$\begin{align}E[E[Y|X=x]] &= \int  E[Y|X=x] \cdot p(X=x)  dx = \iint y \cdot p(Y=y|X=x) p(X=x)dy  dx\\
&= \iint y p(X=x, Y=y) dx dy\\
&= \int y \left (\int p(x, y) dx\right ) dy \\
&= \int y p(y) dy = E[Y]\end{align}$$



"""

# ╔═╡ edd96a1b-b724-48c0-8d1e-2a1862e40ce3
md"""
### Question 3

For linear regression, the model can be written as 

$$y = \boldsymbol{x}^\top \boldsymbol{\beta} + \xi, \;\;\text{where}\;\; \xi \sim N(0, \sigma^2),$$

Given data $\boldsymbol{y} = [y_1, y_2, \ldots, y_n]^\top$ and $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n]^\top$ (the data is given below);

*  what is $E[y|\boldsymbol{x}]$? 
*  what are the parameters of the model ?
*  what is the likelihood function ?
*  maximise the log-likelihood function by (you should implement the algorithm in Julia)
  * normal equation or closed form solution
  * gradient descent
  * Newton's method (check this *https://studres.cs.st-andrews.ac.uk/2020_2021/CS5014/Lectures/CS5014_L06_Logistic_regression_and_newton_method.pdf* if you do not know Newton's method)
*  is the MLE a random variable ? 
  * if so, where the "randomness" comes from ? 
  * what is its distribution ? 
"""

# ╔═╡ 6b2b1d16-7977-4087-bb9e-e1555039a8a3
begin
	dim₃ = 5
	n₃ = 100
	β₃ = rand(dim₃)
	σ² = 0.5
	X = rand(n₃,dim₃)
	X[:,1] .= 1
	y₃ = rand.(Normal.(X * β₃, sqrt(σ²)))
end

# ╔═╡ c7d2106c-4c38-42f3-948a-b21df2f9ca70
md"""

### Solution
"""

# ╔═╡ 5e83c869-894a-4662-ad82-82434445441d
md"

The conditional expectation: 
$$E[y|x] = E[x^\top \beta +\xi|x] = E[x^\top \beta|x] + E[\xi|x]= x^\top \beta + E[\xi] = x^\top \beta + 0 =  x^\top \beta.$$

This is called the regression function. 


$E[\xi] = \int \xi \cdot N(\xi; 0, \sigma^2) d\xi =0$

"

# ╔═╡ 0604d149-c84f-4f2b-a312-b4ba897cb553
md"""

$$p(\boldsymbol y|\boldsymbol X, \boldsymbol \beta, \sigma^2)=\prod_{i=1}^n p(y_i|\boldsymbol x_i, \boldsymbol \beta, \sigma^2) = \prod_{i=1}^n  N(y_i|\boldsymbol x_i^\top \boldsymbol \beta, \sigma^2)$$

"""

# ╔═╡ aae07be9-0cea-403a-b8d4-9dc733f0224d
md"""

**Gradient and hessian**
I am going use multivariate Gaussian distribution on vector $\boldsymbol{y}$ direclty. The likelihood function is

$$l(\boldsymbol{\beta}, \sigma^2) = P(\boldsymbol{y}|\boldsymbol{X}, \boldsymbol{\beta}, \sigma^2)= N(\boldsymbol{y}; \boldsymbol{X\beta}, \boldsymbol{\Sigma}),$$

where $\boldsymbol{\Sigma} = \sigma^2 \boldsymbol{I}_n.$ You should convince yourself why it is the case.

The log likelihood function is then 

$$\begin{align}L(\boldsymbol{\beta}, \sigma^2) &= \ln l(\boldsymbol{\beta}, \sigma^2) = \ln N(\boldsymbol{y}; \boldsymbol{X\beta}, \boldsymbol{\Sigma})\\
&=\ln \left ( (2\pi)^{-\frac{n}{2}}|\boldsymbol{\Sigma}|^{-\frac{1}{2}} \exp \left( -\frac{1}{2} (\boldsymbol{y} - \boldsymbol{X\beta})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{y} - \boldsymbol{X\beta})\right)\right ) \\
&= -\frac{n}{2} \ln 2\pi -\frac{1}{2} \log|\boldsymbol{\Sigma}| -\frac{1}{2} (\boldsymbol{y} - \boldsymbol{X\beta})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{y} - \boldsymbol{X\beta})\end{align}$$


There are occasions where we assume the observation variance $\boldsymbol{\Sigma}$ is known (e.g. time series regression with autoregressive errors and weighted linear regression where we want to discount some noisy observations). Then we only need to take derivative w.r.t $\boldsymbol{\beta}.$

$$\begin{align}\frac{\partial L}{\partial \boldsymbol{\beta}} &= - \frac{1}{2} \cdot 2 (\boldsymbol{y} - \boldsymbol{X\beta})^\top\boldsymbol{\Sigma}^{-1} (-\boldsymbol{X})\\
&= (\boldsymbol{y} - \boldsymbol{X\beta})^\top\boldsymbol{\Sigma}^{-1} \boldsymbol{X}\end{align}$$

Drew note: the gradient dimensions is 1xD. This makes sense because y is a 1xN row vector. The Σ⁻¹ is an nxn matrix, and $\boldsymbol{X}$ is an nxd matrix.

The Hessian is:

$$\boldsymbol{H} = \frac{\partial^2 L}{\partial \boldsymbol{\beta}\partial \boldsymbol{\beta}^\top} = \boldsymbol{X}^\top \boldsymbol{\Sigma}^{-1}(-\boldsymbol{X}) = - \boldsymbol{X}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{X}.$$

Set the gradient to zero, we can find the closed form solution to the optimisation problem:

$$(\boldsymbol{y} - \boldsymbol{X\beta})^\top\boldsymbol{\Sigma}^{-1} \boldsymbol{X} = \boldsymbol{0}^\top$$
$$\Rightarrow \boldsymbol{X}^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{y} - \boldsymbol{X\beta})  = \boldsymbol{0}$$
$$\Rightarrow \hat{\boldsymbol{\beta}}  = (\boldsymbol{X}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{X})^{-1} (\boldsymbol{X}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{y})$$

Substitute in $\boldsymbol{\Sigma} = \sigma^2 \boldsymbol{I}_n$, we have the normal normal equation.

$$\hat{\boldsymbol{\beta}}  = (\boldsymbol{X}^{\top}(\sigma^2 \boldsymbol{I}_n)^{-1}\boldsymbol{X})^{-1} (\boldsymbol{X}^{\top}(\sigma^2 \boldsymbol{I}_n)^{-1}\boldsymbol{y})= (\boldsymbol{X}^{\top}\boldsymbol{X})^{-1} \boldsymbol{X}^{\top}\boldsymbol{y}$$


We need to take the derivative of $L$ w.r.t $\sigma^2$ to find the ML estimator for $\sigma^2$. After a few steps (skipped here), you should find the following estimator

$\hat{\sigma^2}= \frac{1}{n} (\boldsymbol{y}- \boldsymbol{X}\hat{\boldsymbol{\beta}})^\top (\boldsymbol{y}- \boldsymbol{X}\hat{\boldsymbol{\beta}}) = \frac{1}{n} \sum_{i=1}^{n} (y_i-  \boldsymbol{x}_i^\top\hat{\boldsymbol{\beta}})^2$
"""



# ╔═╡ 36227e19-28b2-4710-8075-6afeb143a617
md"""

**Newton's method:**
Newton's step is at $x_t$, update $x_t$ to $x_{t+1}$ according to 

$$x_{t+1} \leftarrow x_t - H(x_t)^{-1} g(x_t)$$

Assume we start from any point $\boldsymbol{\beta}_0 \in R^n$, the first Newton step would be 

$$\boldsymbol{\beta}_{1} \leftarrow \boldsymbol{\beta}_0 - H(\boldsymbol{\beta}_0)^{-1} g(\boldsymbol{\beta}_0),$$

Sub in the definition of the gradient and Hessian, we have

$$\boldsymbol{\beta}_{1} \leftarrow \boldsymbol{\beta}_0 - (-\boldsymbol{X}^\top(\sigma^2 \boldsymbol{I})^{-1}\boldsymbol{X})^{-1} ((\boldsymbol{y} - \boldsymbol{X\beta}_0)^\top(\sigma^2 \boldsymbol{I})^{-1}\boldsymbol{X})^\top = (\boldsymbol{X}^\top \boldsymbol{X})^{-1} \boldsymbol{X}^\top \boldsymbol{y},$$

Newton's method will converge at the first iteration. You should verify the above yourself! The reason is the loss function (or log likelihood function) is a quadratic function, its Taylor's second order approximation is exact: Newton's method is optimising the exact quadratic function.
"""

# ╔═╡ cfdb2c08-bad5-4b6b-9ed9-fe81a8ff7fa4
md""" 
**Sampling distribution of the MLE**

The MLE is $\hat{\boldsymbol{\beta}} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{y}$, it is a random variable from frequentist's perspective: where observations $\boldsymbol{y}$ is assumed random (Bayesians believe the parameters are random variables but not the data). 

Note that $\boldsymbol{y}$ is a Gaussian distributed with $\boldsymbol{\mu} = \boldsymbol{X\beta}$ and covariance $\boldsymbol{\Sigma}= \sigma^2 \boldsymbol{I}_n$ or 

$$p(\boldsymbol{y}|\boldsymbol{X}, \boldsymbol{\beta}, \sigma^2) = N(\boldsymbol{X\beta}, \sigma^2 \boldsymbol{I}_n).$$

There is an important property for Gaussian random variable: all linear combination of Gaussians are still Gaussian. That is if 

$$\boldsymbol{z}\sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma}),$$ then $\boldsymbol{y}=\boldsymbol{Az}$ is also Gaussian distributed with

$$\boldsymbol{y}\sim N(\boldsymbol{A\mu}, \boldsymbol{A\Sigma A}^\top).$$


So $\hat{\boldsymbol \beta}$ is also Gaussian distributed as it is a linear combinations of Gaussian (with $A= (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top$ ). Therefore, we have $\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\mu}_{\hat{\beta}}, \boldsymbol{\Sigma}_{\hat{\beta}})$, where

$\boldsymbol{\mu}_{\hat{\beta}} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{\mu} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{X\beta}= \boldsymbol{\beta}$
and 

$\boldsymbol{\Sigma}_{\hat{\beta}} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{\Sigma} ((\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top)^\top = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \sigma^2\boldsymbol{I}_m ((\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top)^\top = \sigma^2 (\boldsymbol{X}^\top \boldsymbol{X})^{-1}.$

Note its relationship to the Hessian: the sampling distribution's variance is the negative Hessian's inverse.

It is called the **sampling distribution** of the ML estimator. It means if you repeat the same procedure a lot of times, say 2000 times: i.e. collect $n$-observation sample of $X,y$ and calculate its ML estimator. If you collect all 2000 ML estimators, the distribution will be exactly the sampling distribution, a multivariate Gaussian with prescribed mean and covariance.

We can empirically verify the claim by simulation. The following method simulates random observation $\boldsymbol{y}$ based on the linear regression model, then calculates and returns the ML estimator. We repeat the process $mc$ times to obtain supposed samples from the sampling distribution.
"""

# ╔═╡ beddade8-a365-4e5c-9fc2-25ae0fd3d980
begin		
	# Samples y_ vector from normal distribution using input β and σ², then returns the MLE vector.
	function mlSample(β, σ², X)
		y_ = rand.(Normal.(X * β, sqrt(σ²)))
		return (X'*X)^(-1) * X'* y_
	end
	
	# Number of repetitions.
	mc = 2000

	# Initialise matrix.
	mlSamples = zeros(mc, dim₃)
	for i in 1:mc
		# Populate the sample matrix by getting MLE for mc times.
		mlSamples[i,:] = mlSample(β₃, σ², X)
	end

	# Return the mean and covariance.
	(mean(mlSamples, dims=1), cov(mlSamples))
end

# ╔═╡ b2429c76-d25e-49d9-a5ee-0f15798ff819
md"""
The following is the theoretical sampling distribution where we have used the ground true value of $\sigma^2.$
"""

# ╔═╡ a45af332-4a0d-46a3-b2a2-938945130451
(β₃, σ² * (X'*X)^(-1))

# ╔═╡ ec25f8fa-e404-4305-b0d3-290d7cd0a665
md"The following is the scatterplot of the theoretical sampling distribution"

# ╔═╡ bb51e58b-4d62-49ff-abb3-84e7266cea22
cornerplot(rand(MvNormal(β₃, Symmetric(σ² * (X'*X)^(-1))), mc)', compact=true)

# ╔═╡ 5eb06d68-b622-4a9b-a924-7977c9022c24
md"""
The following are the scatterplots of the simulated sampling distribution. 
"""

# ╔═╡ c114437e-ed95-431f-a468-d05454877591
cornerplot(mlSamples, compact=true)

# ╔═╡ 37a4edcf-4ab3-4973-b08b-fceb69b94734
md"""
**Understanding frequentist inference results**
The following table is a standard regression analysis table you would get for basically any statistical analysis software. Pay attention to the standard error column. They are just the diagonal entries of $\boldsymbol{\Sigma}_{\hat{\beta}}=\sigma^2 (\boldsymbol{X}^\top \boldsymbol{X})^{-1}$, where $\sigma^2$ (usually not known) is substituted by a (bias corrected) estimator.
"""

# ╔═╡ 1e5be567-c3d9-4a5e-9f1f-de2d75afe65c
begin
	dfQ3 = DataFrame([X[:,2:end] y₃], :auto)
	rename!(dfQ3,:x5 => :y)
	# glm(@formula(y ~x1+x2+0), dfQ4, Binomial(), LogitLink())
	lm(@formula(y~ x1+x2+x3+x4), dfQ3)
end

# ╔═╡ 97018573-a4af-4a1e-90c0-bc130b133f4e
md"**Biased corrected estimator for $\sigma^2$** The MLE for $\sigma^2$ is biased. Its unbiased estimator is 

$$\hat{\sigma}^2 = \frac{1}{n-d} \sum_{i=1}^n (y_i- \hat y_i^2)$$"

# ╔═╡ a64b005a-80a7-4ec3-9b8d-a84a5501a89b
σₘₗ² = sum((y₃ - X*(X'*X)^(-1) * X'* y₃).^2) /(n₃-dim₃)

# ╔═╡ b6bcf673-ea26-44cd-b69c-2cc14d056fb9
md"Our MLE's variance: "

# ╔═╡ 4a7e462e-daa5-44f2-a8bf-6084ef5a4350
sqrt.(diag(σₘₗ² * (X'*X)^(-1)))

# ╔═╡ 2b2dafde-462b-4784-943d-dc2f590bc857
md"""
### Question 4

For logistic regression, the model can be written as 

$$P(y|\boldsymbol{x}) = \text{Bernoulli}(\sigma(\boldsymbol{x}^\top \boldsymbol{\beta})),$$
Given data $\boldsymbol{y} = [y_1, y_2, \ldots, y_n]^\top$ where $y_i \in \{0, 1\}$ and $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n]^\top$;

*  what is $E[y|\boldsymbol{x}]$ ? Plot $E[y|\boldsymbol{x}]$ by 3-d surface or 2-d contour plot.
*  what are the paremeter for this model ?
*  what is the likelihood function ?
*  simulate $\boldsymbol{y}$ based on the given model ($\boldsymbol{\beta}$ and $\boldsymbol{X}$ are already given below)
*  maximise the log-likelihood function by (you should implement the algorithm in Julia)
  * gradient descent
  * Newton's method
"""

# ╔═╡ a646e1b0-f7ab-43e2-bd47-aaab18a892f2
begin
# 	number of dimensions 
	dim₄ = 2
# 	number of observations
	n₄ = 100
# 	a n*m matrix, for this case there is no intercept term, or equivalently $\beta_0 = 0$
	X₄ = rand(n₄,dim₄)
# 	make X₄ ranging between -10 to 10
	X₄ = X₄ .* 20 .- 10.0
	β₄ = rand(dim₄)	
end

# ╔═╡ 43304240-1a9a-445b-bc90-4bd0c1c0b0d0
md"""
### Solution

First, let's see what the regression function looks like
"""

# ╔═╡ afa92123-7c04-48e7-a582-a7671bcdf497
begin	
	logReg(x, β) = logistic(x' * β)	
end

# ╔═╡ c3e609cd-f3ef-47ee-8283-8bd50d430e5c
begin
	x1 = range(-10, stop=10, length=100)
	x2 = range(-10, stop=10, length=100)
	β_ = [1,1.] .* 1.
	# β_ = β₄
	plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], β_), st=:surface)
end

# ╔═╡ a6adc5b4-5fb6-48aa-bf33-e0702988841b
plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], β_), st=:contour)

# ╔═╡ b1f96d2f-3595-4863-9b30-a22de2536202
md"""
We will first define the log likelihood function; we can either use Julia's builtin Bernoulli r.v. or define our own.

The log likelihood function is 

$$\begin{align} L(\beta) &= \ln\prod_{i=1}^n \text{Bernoulli}(y_i;\sigma(x_i^\top\beta)) \\
&= \sum_{i=1}^n \ln \left (\sigma(x_i^\top\beta)^{y_i} (1-\sigma(x_i^{\top} \beta))^{1-y_i}\right ) \\
&= \sum_{i=1}^n \underbrace{y_i \ln(\sigma(x_i^\top \beta)) + (1-y_i) \ln(1-\sigma(x_i^\top \beta))}_{L_i(\beta)}\end{align}$$
"""

# ╔═╡ 649b7155-4c2f-4ff7-a1bd-c82686ecb687
begin
# 	Firstly, we simulate the data; here I have used the Julia's built in function
	y₄ = Int.(rand.(Bernoulli.(logistic.(X₄ * β₄))))
#   You can also simply use the following code; each toss yᵢ is the result of tossing a bent coin with p=σᵢ 	
    # y₄ = Int.(rand(n₄) .< logistic.(X₄ * β₄))
	
#   Next we define the log likelihood function		
	function logRegLogLik(y, X, β)
		σ = logistic.(X*β)
		sum(logpdf.(Bernoulli.(σ), y))
	end
	
#   Again we can define our own
# 	y*log(σ) + (1-y)*log(1-σ)
	function mylogRegLogLik(y, X, β)
		σ = logistic.(X*β)
# 		using log directly is not numerically stable! e.g when σ = 0 or 1, log(0) becomes -inf
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
	
# 	we can of course check whether the two are the same
	randβ = rand(dim₄)
	mylogRegLogLik(y₄, X₄, randβ) - logRegLogLik(y₄, X₄, randβ)
end

# ╔═╡ e81a1454-4faf-41b2-a903-713be2321c2a
y₄

# ╔═╡ 480bd22d-cc16-4d90-a0c4-b50e6558d315
md"""

The log likelihood function is 


$$\begin{align} L(\beta) &= \ln \prod_{i=1}^n P(y_i|\sigma(x_i^\top\beta)) \\
&= \sum_{i=1}^n \ln \left (\sigma(x_i^\top\beta)^{y_i} (1-\sigma(x_i^{\top} \beta))^{1-y_i}\right ) \\
&= \sum_{i=1}^n \underbrace{y_i \ln(\sigma(x_i^\top \beta)) + (1-y_i) \ln(1-\sigma(x_i^\top \beta))}_{L_i(\beta)}\end{align}$$


Take derivative w.r.t $\beta$


$$\begin{align} \frac{\partial L}{\partial \beta} = \sum_{i=1}^n \frac{\partial L_i}{\partial \beta}\end{align}$$


$$\begin{align} \frac{\partial L_i}{\partial \beta} &= y_i \cdot \sigma_i^{-1} \cdot \frac{\partial \sigma_i}{\partial (x_i^\top \beta)} \cdot x_i + (1-y_i) (1-\sigma_i)^{-1} \cdot (-1) \cdot \frac{\partial \sigma_i}{\partial (x_i^\top \beta)} \cdot x_i \\
&= y_i \cdot \sigma_i^{-1} \cdot \sigma_i(1-\sigma_i)\cdot x_i +(y_i-1)(1-\sigma_i)^{-1}  \cdot \sigma_i(1-\sigma_i) \cdot x_i \\
&= y_i \cdot (1-\sigma_i)\cdot x_i + (y_i-1)  \cdot \sigma_i \cdot x_i \\
&= y_i x_i - y_i \sigma_i x_i + y_i\sigma_i x_i - \sigma_i x_i \\
&= (y_i-\sigma_i)x_i\end{align},$$

where I have used the following property of sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$

$\frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1-\sigma(x))$

Therefore, the gradient is (using column gradient notation):

$$\frac{\partial L}{\partial \boldsymbol{\beta}} = \sum_{i=1}^n (y_i-\sigma_i) x_i = \boldsymbol{X}^\top(\boldsymbol{y}- \boldsymbol{\sigma}),$$ where $\boldsymbol{\sigma} = \begin{bmatrix}\sigma(x_1 ^\top \boldsymbol{\beta})\\ \sigma(x_2 ^\top \boldsymbol{\beta})\\ \vdots\\\sigma(x_n ^\top \boldsymbol{\beta})\end{bmatrix}.$

"""

# ╔═╡ 74bb2923-c286-46b6-bbae-4ebe6d46b84d
md"""

We shall continue to find the Hessian matrix. By definition, Hessian matrix entries are second derivatives. 

$$H = \left (\frac{\partial^2 L}{\partial \beta_i \partial \beta_j}\right )_{i,j}$$


We just need to find $\frac{\partial^2 L}{\partial \beta_i \partial \beta_j}$ for a particular pair of $i,j \in \{1,2,\ldots, d\}$ ($d$ is the dimension of $\beta$). 

We already know 

$$\frac{\partial L}{\partial \boldsymbol{\beta}} = \sum_{i=1}^n (y_i-\sigma_i) x_i = \boldsymbol{X}^\top(\boldsymbol{y}- \boldsymbol{\sigma}),$$ 

The $j$-th partial derivative is just the $j$-th entry of the gradient

$$\frac{\partial L}{\partial \beta_j} = \left [\frac{\partial L}{\partial \boldsymbol{\beta}}\right ]_j = \tilde{\boldsymbol{x}}_j^\top (\boldsymbol{y} - \boldsymbol{\sigma})=\begin{bmatrix} x_{1,j} & x_{2,j} & \ldots & x_{n,j}\end{bmatrix} \begin{bmatrix} y_1 - \sigma(\boldsymbol{x}_1^\top \boldsymbol{\beta})\\ y_2 - \sigma(\boldsymbol{x}_2^\top \boldsymbol{\beta})\\ \vdots \\y_n - \sigma(\boldsymbol{x}_n^\top \boldsymbol{\beta})\end{bmatrix},$$ where $\tilde{\boldsymbol{x}}_j$ is the $j$-th column of $X$, or the $j$-th feature across all $n$ observations. 

$$\frac{\partial^2 L}{\partial \beta_i \partial \beta_j} = \frac{\partial }{\partial \beta_i}\frac{\partial L}{\partial \beta_j} =\sum_{k=1}^n \frac{\partial (x_{k,j}\cdot (y_k - \sigma(\boldsymbol{x}_k^\top \boldsymbol{\beta})))}{\partial \beta_i} = \sum_{k=1}^n -x_{k,j} \cdot \sigma_k (1-\sigma_k) \cdot x_{k,i},$$

which can be compactly written as $\boldsymbol{X}^\top \boldsymbol{D} \boldsymbol{X},$ where $\boldsymbol{D} = \text{diag} (\sigma_k(\sigma_k-1))$ for $k =1,\ldots,n$.

To see this, note that $(\boldsymbol{X}^\top \boldsymbol{X})_{i,j} = \sum_{k=1}^n x_{k,i}x_{k,j}$ i.e. inner product between $i,j$-th features across $n$ observations. By multiplying a diagonal matrix in between, we multiply each product item a weight (with the corresponding diagonal entry of $\boldsymbol{D}$).

"""

# ╔═╡ 28088880-0833-48c6-b418-d5a26232c0cc
begin
	
	function logRegLogLiks(y, X, β)
		σ = logistic.(X*β)
		grad = X' * (y - σ)
		H = X' * (σ .* (σ .-1) .* X)
# 		you may also write it as derived by creating a diagonal matrix D
		# H = X' * diagm(σ .* (σ .-1)) * X)
		sum(logpdf.(Bernoulli.(σ), y)), grad, H
	end	
	
end

# ╔═╡ 93c45649-f984-4668-b101-f978ac3cbd89
md"""

Before using your derived gradient and Hessian, you should always test them! Below I have used Julia's Forward Diff (an auto differentiation package) and Finite Differences (a gradient approximation method) to check my answer. 
"""

# ╔═╡ f570537a-71c2-4c44-b8c9-f381e52b60a3
begin
	β_test = rand(dim₄)
# 	use forward diff to check
	ForwardDiff.gradient((x) -> logRegLogLiks(y₄, X₄, x)[1], β_test) - logRegLogLiks(y₄, X₄, β_test)[2]
	ForwardDiff.hessian((x) -> logRegLogLiks(y₄, X₄, x)[1], β_test) - logRegLogLiks(y₄, X₄, β_test)[3]
	
#   use finite difference method to check	
	grad(central_fdm(5,1), (x) -> logRegLogLiks(y₄, X₄, x)[1], β_test)[1]- logRegLogLiks(y₄, X₄, β_test)[2]
end

# ╔═╡ 88ddc046-5a95-4cb6-a183-dd5cbebb91a3
begin
	β_test2 = rand(dim₄)
	# 	use forward diff to check
	ForwardDiff.gradient((x) -> logRegLogLiks(y₄, X₄, x)[1], β_test2)
end

# ╔═╡ 9230c181-9bdc-4006-b984-7eee63f72037
md"""
After deriving the gradient and Hessian, we are ready to optimise the log likelihood function. We will use both gradient descent and Newton's method. 
"""

# ╔═╡ 32f95a2f-b615-480a-a4a1-3c95eda00d7c
begin

	#ftt is just the likekihood?
	function mleGradAscent(llFuns, x0; tol = 1e-5, maxIters=10000, λ= 0.001)
		x = x0
		ftt,_,_ = llFuns(x)
		fs = zeros(maxIters)
		fs[1] = ftt
		t = 2
		while t <= maxIters
			ft, gt, _ = llFuns(x)
			x = x + λ*gt
			fs[t] = llFuns(x)[1]
			if abs(fs[t]- fs[t-1]) < tol
				t= t+1
				break
			end
			t = t+1
		end
		return x, fs[1:t-1]
	end
	
	
	function mleNewton(llFuns, x0; tol = 1e-5, maxIters=1000, regularise=false, linesearch=false)
		x = x0
		ftt,_,_ = llFuns(x)
		fs = zeros(maxIters)
		fs[1] = ftt
		t = 2
		while t <= maxIters
			ft, gt, Ht = llFuns(x)
# 			to make sure Ht is invertible (or to improve numerical stability), we usually add small constants to the diagonal entries of Ht
			if regularise
 				Ht[diagind(Ht)] = diag(Ht) .+ 1e-4
			end
# 			λ usually set by a crude line search to avoid accdidentally minimising a function!
			dt = Ht\gt
			λts = collect(0.1:0.1:1.0)
			maxF = -Inf
			maxx = x - dt
			if linesearch
				for i in 1:length(λts)
					xt = x - λts[i] * dt
					fxt,_,_ = llFuns(xt)
					if(fxt > maxF)
						maxF = fxt
						maxx = xt 
					end
				end
			end
			x = maxx
			fs[t] = llFuns(x)[1]
			if abs(fs[t]- fs[t-1]) < tol
				t= t+1
				break
			end
			t = t+1
		end
		return x, fs[1:t-1]
	end
	
end

# ╔═╡ 32598da5-e198-45fd-82aa-32111d8cae1e
md"""
I use Julia's GLM as a reference to check my solution.
"""

# ╔═╡ 2f27e460-9f27-4fce-bc87-4b58ba2dcfbd
begin
	dfQ4 = DataFrame([X₄ y₄], :auto)
	rename!(dfQ4,:x3 => :y)
	glm(@formula(y ~x1+x2+0), dfQ4, Binomial(), LogitLink())
end

# ╔═╡ 0220e2a2-1119-4aa5-ac43-bc37c3e61d3f
begin	
	llfQ4(x) = logRegLogLiks(y₄, X₄, x)
	βq4gd, fs=mleGradAscent(llfQ4, zeros(dim₄); tol = 1e-5, maxIters=10000, λ= 0.001)
# plot(fs)
end

# ╔═╡ eea9ea11-3582-4890-8bff-2e01b2ddc759
plot(fs)

# ╔═╡ a6d35376-b3e6-4b42-8862-a11c5da0a174
β̂_nt, ll_nt = mleNewton(llfQ4, zeros(dim₄); tol = 1e-5, maxIters=1000, regularise=false, linesearch=false)

# ╔═╡ c3d9131a-e150-4b5e-925a-6450477ae89c
plot(ll_nt)

# ╔═╡ 3abace52-fecd-4f12-aad5-5d239fdd7774
md"""
Note the difference between Newton's method and Gradient ascent. Newton's method converges much faster: 4-5 iterations!

"""

# ╔═╡ 2bd64856-9f58-44d5-af7a-972c91878c9c
md"""

## Extra exercises*
You can try the following to test your understanding after the meeting.

### Question 4

For Poisson regression (where observations $y$ are usually counting data, say web page visitor counts etc.), the model can be written as 

$$P(y|\boldsymbol{x}) = \text{Poisson}(\exp(\boldsymbol{x}^\top \boldsymbol{\beta})),$$
Given data $\boldsymbol{y} = [y_1, y_2, \ldots, y_n]^\top$ where $y_i \in \{0, 1, 2,\ldots\}$ and $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n]^\top$;
*  What is $E[y|\boldsymbol{x}]$ ?
*  What are the paremeter for this model ?
*  What is the likelihood function ?
*  Simulate $\boldsymbol{y}$ based on the given model 
*  Maximise the log-likelihood function by (implement the algorithm in Julia)
  * gradient descent
  * Newton's method
"""

# ╔═╡ 7c5a6de4-c194-4628-a891-82d56f9901f1
begin
	# 	number of dimensions 
	dim₅ = 5
# 	number of observations
	n₅ = 100
# 	a n*m matrix, for this case there is no intercept term, or equivalently $\beta_0 = 0$
	X₅ = rand(n₅,dim₅) *2 .-1
	β₅ = rand(dim₅)	
end

# ╔═╡ 72b406db-531a-4b2c-ad64-4bb0b5814111
md"""
### Question 5

For Geometric regression (where observations $y$ are usually counting data, say how many times you need to try until success), the model can be written as 

$$P(y|\boldsymbol{x}) = \text{Geometric}(\sigma(\boldsymbol{x}^\top \boldsymbol{\beta})),$$
Given data $\boldsymbol{y} = [y_1, y_2, \ldots, y_n]^\top$ where $y_i \in \{0, 1, 2,\ldots\}$ and $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n]^\top$;
*  What is $E[y|\boldsymbol{x}]$ ?
*  What are the paremeter for this model ?
*  What is the likelihood function ?
*  Simulate $\boldsymbol{y}$ based on the given model 
*  Maximise the log-likelihood function by (implement the algorithm in Julia)
  * gradient descent
  * Newton's method
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DataFrames = "~1.3.4"
Distributions = "~0.25.16"
FiniteDifferences = "~0.12.24"
ForwardDiff = "~0.10.30"
GLM = "~1.8.0"
Plots = "~1.22.3"
StatsFuns = "~0.9.18"
StatsPlots = "~0.14.28"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e8a30e8019a512e4b6c56ccebc065026624660e8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "29890dfbc427afa59598b8cfcc10034719bd7744"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.6"

[[FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "0ee1275eb003b6fc7325cb14301665d1072abda1"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.24"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "039118892476c2bf045a43b88fcb75ed566000ff"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "c6cf981474e7094ce044168d329274d797843467"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.6"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "9d8c00ef7a8d110787ff6f170579846f776133a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.4"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "cfbd033def161db9494f86c5d18fbf874e09e514"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.3"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "4352d5badd1bc8bf0a8c825e886fa1eda4f0f967"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.30"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═97c8fe8a-21c0-11ec-1100-638723043ca3
# ╟─b1d6136d-403d-47b7-aa9d-60b59d0b15b6
# ╟─7da12ea5-d615-4de1-a728-32c2e5980432
# ╟─14f6f12b-096e-450b-bc42-4bce785e686c
# ╟─e67efa89-9b1c-44f8-a933-117592524609
# ╟─edd96a1b-b724-48c0-8d1e-2a1862e40ce3
# ╠═6b2b1d16-7977-4087-bb9e-e1555039a8a3
# ╟─c7d2106c-4c38-42f3-948a-b21df2f9ca70
# ╟─5e83c869-894a-4662-ad82-82434445441d
# ╟─0604d149-c84f-4f2b-a312-b4ba897cb553
# ╟─aae07be9-0cea-403a-b8d4-9dc733f0224d
# ╟─36227e19-28b2-4710-8075-6afeb143a617
# ╟─cfdb2c08-bad5-4b6b-9ed9-fe81a8ff7fa4
# ╠═beddade8-a365-4e5c-9fc2-25ae0fd3d980
# ╟─b2429c76-d25e-49d9-a5ee-0f15798ff819
# ╠═a45af332-4a0d-46a3-b2a2-938945130451
# ╟─ec25f8fa-e404-4305-b0d3-290d7cd0a665
# ╠═bb51e58b-4d62-49ff-abb3-84e7266cea22
# ╟─5eb06d68-b622-4a9b-a924-7977c9022c24
# ╠═c114437e-ed95-431f-a468-d05454877591
# ╟─37a4edcf-4ab3-4973-b08b-fceb69b94734
# ╠═1e5be567-c3d9-4a5e-9f1f-de2d75afe65c
# ╟─97018573-a4af-4a1e-90c0-bc130b133f4e
# ╠═a64b005a-80a7-4ec3-9b8d-a84a5501a89b
# ╟─b6bcf673-ea26-44cd-b69c-2cc14d056fb9
# ╠═4a7e462e-daa5-44f2-a8bf-6084ef5a4350
# ╟─2b2dafde-462b-4784-943d-dc2f590bc857
# ╠═a646e1b0-f7ab-43e2-bd47-aaab18a892f2
# ╟─43304240-1a9a-445b-bc90-4bd0c1c0b0d0
# ╠═afa92123-7c04-48e7-a582-a7671bcdf497
# ╠═c3e609cd-f3ef-47ee-8283-8bd50d430e5c
# ╠═a6adc5b4-5fb6-48aa-bf33-e0702988841b
# ╟─b1f96d2f-3595-4863-9b30-a22de2536202
# ╠═649b7155-4c2f-4ff7-a1bd-c82686ecb687
# ╠═e81a1454-4faf-41b2-a903-713be2321c2a
# ╟─480bd22d-cc16-4d90-a0c4-b50e6558d315
# ╟─74bb2923-c286-46b6-bbae-4ebe6d46b84d
# ╠═28088880-0833-48c6-b418-d5a26232c0cc
# ╟─93c45649-f984-4668-b101-f978ac3cbd89
# ╠═f570537a-71c2-4c44-b8c9-f381e52b60a3
# ╠═88ddc046-5a95-4cb6-a183-dd5cbebb91a3
# ╟─9230c181-9bdc-4006-b984-7eee63f72037
# ╠═32f95a2f-b615-480a-a4a1-3c95eda00d7c
# ╟─32598da5-e198-45fd-82aa-32111d8cae1e
# ╠═2f27e460-9f27-4fce-bc87-4b58ba2dcfbd
# ╠═0220e2a2-1119-4aa5-ac43-bc37c3e61d3f
# ╠═eea9ea11-3582-4890-8bff-2e01b2ddc759
# ╠═a6d35376-b3e6-4b42-8862-a11c5da0a174
# ╠═c3d9131a-e150-4b5e-925a-6450477ae89c
# ╟─3abace52-fecd-4f12-aad5-5d239fdd7774
# ╟─2bd64856-9f58-44d5-af7a-972c91878c9c
# ╠═7c5a6de4-c194-4628-a891-82d56f9901f1
# ╟─72b406db-531a-4b2c-ad64-4bb0b5814111
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
