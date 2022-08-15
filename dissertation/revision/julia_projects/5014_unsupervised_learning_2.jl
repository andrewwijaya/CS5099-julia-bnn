### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 566619ff-f7cd-44c3-8620-58544fb8492f
begin
	using Plots, Distributions,Random, StatsFuns, StatsBase, Clustering, LinearAlgebra
	using PlutoUI
	using StatsPlots
end

# ╔═╡ d5f4454a-997f-11ec-215d-2f93f237aa1b
md"""
##### Initializing packages

*When running this notebook for the first time, this could take up to 10 minutes. Hang in there !*
"""

# ╔═╡ 3a68f4fd-c333-4293-b1b4-bff8eff99409
Random.seed!(123);

# ╔═╡ 2437de37-77ad-453c-b974-9a31549938b7
md"""

> I recommend you download and run this Pluto notebook (on studres) when you study or review the lecture. A quick reference for Julia can be found here: [minimum starting syntax](https://computationalthinking.mit.edu/Spring21/basic_syntax/), and [getting-started with Julia](https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started). Note that programming in Julia is not required for this module and Julia programs are only used to render the slides. With that said, it should greatly help you understand the mathematics by checking and understanding the code.
"""

# ╔═╡ 59a76268-f094-4504-bf7c-046f651f956b
html"<button onclick='present()'>present</button>"

# ╔═╡ 56785ca6-3af0-460f-9782-2a1eccaeb1c0
md"""
# CS5014 Machine Learning
#### Lecture 12. EM algorithm (unsupervised learning 2)
###### Lei Fang 
"""

# ╔═╡ 817128a7-a4e6-4616-bd95-33459fa25866
md"""
## Today's topic

* finite mixture of Gaussians
  * choose the cluster size $K$
  * measure clustering performance

* general finite mixture models
  * EM algorithm for general mixture
  
  * application: finite mixture of regressions
* general EM algorithm
  * explain EM
  * see some applications (if we have time)
"""

# ╔═╡ c1f19943-ea5b-4314-b775-841ef6da6d29
md"""

## Some essential probability theory


**Sum rule**:  

$p(x) = \sum_y p(x, y),\;\; p(y)= \sum_{x} p(x, y)$


**Product rule**:

$p(x,y) = p(x)p(y|x) = p(y) p(x|y)$


**Independence** (all are equivalent):

$p(x,y) = p(x)p(y),\;\;p(x|y)=p(x),\;\; p(y|x) =p(y)$


**Conditional independence** (all are equivalent):

$p(x,y|z) = p(x|z)p(y|z),\;\;p(x|y, z)=p(x|y),\;\;p(y|x, z)=p(y|x)$


**Expectation** (not strictly required but good to know it; see Appendix for examples):

$E[f(x)] = \sum_{x} f(x) p(x)$


Replace all summation with integration if $x,y$ are continous r.v.s
"""

# ╔═╡ d96e3952-0d74-4eaf-b614-40ae0f19aa33
md"""
## Recap: multivariate Gaussian


A d-dimensional multivariate Gaussian with mean $\mu$ and covariance $\Sigma$ has density

$$\begin{equation*}
p({x})={N}({x}| {\mu}, {\Sigma}) =  \underbrace{\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}}_{\text{normalising constant}}
\exp \left \{-\frac{1}{2} \underbrace{({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})}_{\text{distance btw } x \text{ and } \mu }\right\}
\end{equation*}$$

* ``\mu \in R^d``: mean  
* ``\Sigma``: covariance matrix, a $d$ by $d$ symmetric and *positive definite* matrix 
* the kernel is a **distance measure** between $x$ and $\mu$

  $({x} - {\mu})^\top \Sigma^{-1}({x}-{\mu})$
  * e.g. when $\Sigma=I$, we recover Euclidean distance
* ``p(x)`` is negatively related to the distance
  * further away ``x`` is from ``\mu``, the smaller probability density ``p(x)``
* such a distance/similarity metaphor applies to almost all distributions (*esp.* location-scale distributions)
"""

# ╔═╡ 31828fb6-4878-44e9-bef8-7b28e910b4f6
md"``\sigma_1^2``: $(@bind σ₁² Slider(0.5:0.5:5, default=1)); ``\sigma_2^2``: $(@bind σ₂² Slider(0.5:0.5:5, default=1)); 

``\sigma_{12}=\sigma_{21}``: $(@bind σ₁₂ Slider(-minimum([σ₁², σ₂²]):0.05:minimum([σ₁², σ₂²]), default=0))"

# ╔═╡ 6f062399-da4e-4a3b-a500-68a6d618a15f
md"``\Sigma=[`` $(σ₁²), $(σ₁₂); $(σ₁₂) $(σ₂²) ``] ``"

# ╔═╡ a918e796-93a7-4f40-aaaf-3e3a80855430
begin
	gr()
	μ₂ = zeros(2)
	x₁s = μ₂[1]-3:0.1:μ₂[1]+3
	x₂s = μ₂[2]-3:0.1:μ₂[2]+3	
	mvnsample = randn(2, 500);
	Σ₃ = [σ₁² σ₁₂; σ₁₂ σ₂²]
	# cholesky decomposition of Σ (only to reuse the random samples)
	L₃ = cholesky(Σ₃).L
	mvn₃ = 	MvNormal(μ₂, Σ₃)
	# μ + L * MvNormal(0, I) = MvNormal(μ, LLᵀ)
	spl₃ = μ₂.+ L₃ * mvnsample
	scatter(spl₃[1,:], spl₃[2,:], ratio=1, label="", xlabel="x₁", ylabel="x₂")	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label="μ", markershape = :diamond, markersize=8)	
	λs, vs =eigen(Σ₃)
	v1 = (vs .* λs')[:,1]
	v2 = (vs .* λs')[:,2]
	quiver!([μ₂[1]], [μ₂[2]],quiver=([v1[1]], [v1[2]]), linewidth=4, color=:red)
	quiver!([μ₂[1]], [μ₂[2]],quiver=([v2[1]], [v2[2]]), linewidth=4, color=:red)
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₃, [x1, x2]), levels=4, linewidth=4, st=:contour)	
end

# ╔═╡ beae3af1-db1f-4a5b-b23d-f6b565028f61
md"""
## Recap: mixture of Gaussian models


**Finite mixture of Gaussians** is just QDA (but with assumption that labels $z^i$ are missing)

The probabilistic graphical model (Bayesian Networks) representation of mixture model and QDA is 
* each node is random variable
* each edge represents the dependence relationship (each ``x_i`` depends on its label ``z^i``):


$p(z^i, x^i) = p(z^i) p(x^i|z^i)$
* then we can specify the (conditional) probability distributions (CPDs) for each node (r.v.)


  * ``z^i:`` a multinoulli r.v. (K facet die): ``p(z^i=k) = \begin{cases}\pi_1 & k=1 \\\pi_2& k=2 \\ \vdots\\ \pi_K & k=K \end{cases}``
  * ``x^i:`` Gaussian distributed depended on the label of $z^i$

$$p(x^i|z^i=k) = N(x^i; \mu_k, \Sigma_k)$$
"""

# ╔═╡ 5950b105-6264-4837-8f5f-8abe78638b59
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5014/mixturepgm.png' width = '900' /></center>"

# ╔═╡ 3d9a9d05-a9ae-4eaa-bd5d-da108ee9c903
md"""

The marginal distribution of $x$ becomes

$$\begin{align}p(x^i) &= \sum_{k=1}^K p(z^i=k, x^i) = \sum_{k=1}^K p(z^i=k)p(x^i|z^i=k)\\
&= \sum_{k=1}^K \pi_k \cdot N(x^i; \mu_k, \Sigma_k)\end{align}$$

* eq 1: sum rule, eq 2: prod rule
* a linear superposition of K distributions ``p(x^i|z^i=k)`` (Gaussian likelihood assumption for $x^i$)
* interpretation: since we do not know which $z^i=k$ is responsible for $x^i$, we add all possibilities up 

## Why marginalisation ?

Why do we need to do the marginalisation?

* because $z$s are not observed, the observed is $D=\{x^i\}$ only!

And **likelihood** is defined as **the conditional probability of observing the (observed) data**

$P(D|\theta) = P(\{x^i\}|\theta) = \prod_{i=1}^n \left(\sum_{k=1}^K \pi_k \cdot N(x^i; \mu_k, \Sigma_k)\right )$
* without marginalisation, we do not even have a likelihood model to work with (optimise)


On the other hand, QDA's likelihood model is 

$P(D|\theta) = P(\{x^i, z^i\}|\theta) = \prod_{i=1}^n p(z^i)p(x^i|z^i)= \prod_{i=1}^n 
 \pi_{z^i} N(x^i; \mu_{z^i}, \Sigma_{z^i})$

Mixture model is just the marginalised version of QDA.
"""

# ╔═╡ 729480eb-18d1-403c-acf8-d4d85882231f
md"""
## An example of mixture of Gaussians

Mixture of (3) Gaussians

$p(x) = \sum_{k=1,2,3} \pi_k N(x; \mu_k, \Sigma_k)$

where
$\pi = [0.25, 0.5, 0.25];$ $\mu_1 = [1 , 1], \Sigma_1 = \begin{bmatrix}1, -0.9\\-0.9, 1\end{bmatrix};$
$\mu_2 = [0 , 0], \Sigma_2 = \begin{bmatrix}1, 0.9\\0.9, 1\end{bmatrix};$
$\mu_3 = [-1 , -1], \Sigma_3 = \begin{bmatrix}1, -0.9\\-0.9, 1\end{bmatrix}$
"""

# ╔═╡ 65c8c34a-575d-4abc-9db7-547baff43897
begin
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 1.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 1.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., -1],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 800
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ 67b0a6ec-0db3-4e1c-a187-147287749c28
md"""

## Effects of ``p(z)=\pi``
"""

# ╔═╡ 9c2c18f4-22ab-4c75-bcc2-7a649778ed0f
md"``p(z)= \pi``"

# ╔═╡ 5011c678-e36d-4fc0-9705-0847823eee7a
md" ``\pi_1\propto`` $(@bind n₁0 Slider(1:50, default=1));	``\pi_2\propto`` $(@bind n₂0 Slider(1:50, default=2)); ``\pi_3\propto`` $(@bind n₃0 Slider(1:50, default=1))"

# ╔═╡ ee03f8bf-042a-4f6a-b64a-12e17013978e
begin
	πs0 = [n₁0, n₂0, n₃0]
	πs0 = πs0/sum(πs0)
end

# ╔═╡ 8b8f7e0b-3c56-4c89-aa72-21465b993ecf
md"""

## EM algorithm for mixture of Gaussians

An iterative method that aims at finding the maximum likelihood estimator of $\theta=\{\pi, \mu_k, \Sigma_k\}_{k=1}^K$

$$\theta = \arg\max_{\theta} \ln p(D|\theta) = \arg\max_{\theta} \ln \prod_{i=1}^n  p(x^i|\theta) = \arg\max_{\theta} \sum_{i=1}^n \ln \left(\sum_{k=1}^K \pi_k N(x;\mu_k, \Sigma_k)\right)$$

* E-step: improvise an estimation of labels

$$I(z^i=k) \Rightarrow p(z^i=k|x^i)$$
* M-step: weighted likelihood estimation 


The algorithm is:

**Initilisation**: random guess $\theta^{(0)} =\{{\pi_k}^{(0)}, \mu_k^{(0)}, \Sigma_k^{(0)}\}_{k=1}^K$


* **Expectation step** (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$w_{ik} \leftarrow p(z^i=k|{x}^i) = \frac{\pi_k^{(t)} N(x^i; {\mu}_k^{(t)}, {\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} N(x^i; {\mu}_j^{(t)}, {\Sigma}_j^{(t)})}$$


* **Maximisation step** (M step): update ${\theta}^{(t)}$, for $k=1\ldots K$

$\pi_k^{(t)} \leftarrow \frac{1}{n}\sum_{i=1}^n w_{ik}$

${\mu}_{k}^{(t)} \leftarrow \frac{1}{\sum_{i=1}^n w_{ik}} \sum_{i=1}^n w_{ik}x^i$

${{\Sigma}}_{k}^{(t)} \leftarrow \frac{1}{\sum_{i=1}^n w_{ik}} \sum_{i=1}^n w_{ik} (x^i-{{\mu}}_{k})(x^i-{{\mu}}_{k})^\top$

$t\leftarrow t+1$
**Repeat** above two steps until converge

"""

# ╔═╡ c235cffa-9612-45b7-aee8-9d4747f761de
function e_step(data, mvns, πs)
	K = length(mvns)
	# logLiks: a n by K matrix of P(dᵢ|μₖ, Σₖ)
	logLiks = hcat([logpdf(mvns[k], data') for k in 1:K]...)
	# broadcast log(P(zᵢ=k)) to each row 
	logPost = log.(πs') .+ logLiks
	# apply log∑exp to each row to find the log of the normalising constant of p(zᵢ|…)
	logsums = logsumexp(logPost, dims=2)
	# normalise in log space then transform back to find the responsibility matrix
	ws = exp.(logPost .- logsums)
	# return the responsibility matrix and the log-likelihood
	return ws, sum(logsums)
end

# ╔═╡ fa05117f-f9c3-4735-a45e-5bff4cbc7447
function m_step(data, ws)
	_, d = size(data)
	K = size(ws)[2]
	ns = sum(ws, dims=1)
	πs = ns ./ sum(ns)
	# weighted sums ∑ wᵢₖ xᵢ where wᵢₖ = P(zᵢ=k|\cdots)
	ss = data' * ws
	# the weighted ML for μₖ = ∑ wᵢₖ xᵢ/ ∑ wᵢₖ
	μs = ss ./ ns
	Σs = zeros(d, d, K)
	for k in 1:K
		error = (data .- μs[:,k]')
		# weighted sum of squared error
		# use Symmetric to remove floating number numerical error
		Σs[:,:,k] =  Symmetric((error' * (ws[:,k] .* error))/ns[k])
	end
	# this is optional: you can just return μs and Σs
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	return mvns, πs[:]
end

# ╔═╡ 4dfbd0d2-1ca2-4495-b387-f1f0aaa9a85c
function em_mix_gaussian(data, K=3; maxIters= 100, tol= 1e-4, init_step="e")
	# initialisation
	n,d = size(data)
	if init_step == "e"
		zᵢ = rand(1:K, n)
		μs = zeros(d, K)
		[μs[:,k] = mean(data[zᵢ .== k,:], dims=1)[:] for k in 1:K] 
	elseif init_step == "m"
		μs = data[rand(1:n, K), :]'
	else
		μs = randn(d,K)
		μs .+= mean(data, dims=1)[:] 
	end
	Σs = zeros(d,d,K)
	Σs .= Matrix(1.0I, d,d)
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	πs = 1/K .* ones(K)
	zs = zeros(n,K)
	logLiks = Array{Float64,1}()
	i = 1
	for i in 1:maxIters
		# E-step
		zs, logLik = e_step(data, mvns, πs)
		# M-step
		mvns, πs = m_step(data, zs)
		push!(logLiks, logLik)
		# be nice, let it run at least three iters
		if i>2 && abs(logLiks[end] - logLiks[end-1])< tol
			break;
		end
	end
	return logLiks, mvns, πs, zs
end

# ╔═╡ 17445efc-3733-44ed-bc7a-6e2a36efa63b
md"""
## Local optimum

The likelihood function can be very complicated and non-concave 
* multiple local maximum

EM might get trapped at some local optimum if a bad intialisation is used
* e.g. one extreme initialisation example: all $z^i=1$; i.e. all data assigned to one cluster
  * mixture collapses to one singular Gaussian
* no improvement can be made even in the first iteration
  
Solution: repeat the algorithm a few times with different random initialisations
* and use likelihood as a guide to find the best model


"""

# ╔═╡ 3be5f4f5-9e1e-46b0-8b34-551d7438ab0d
md"""
## Demonstration on EM algorithm

"""

# ╔═╡ 983e0e36-cadc-428a-8eb2-06d710fc300a
md"""

## Revisit K-means

K-means is a specific case of EM with the following assumptions

Model wise, the underlying model is a restrictive mixture model 

$p(x) = \sum_{k=1}^K \frac{1}{K} N(x; \mu_k, I)$

* the prior is uniform distributed $$p(z^i=k) = \pi_k = 1/K$$
* and covariances are tied but also fixed to be identity matrix $\Sigma_k = I$, which explains the Euclidean distance used

**assignment step** is just a **hard E step** (winner takes all)

$$w_{ik} \leftarrow \begin{cases} 1, \text{ if } k=\arg\max_{k'} p(z^i=k'|{x}^i)& \\ 0, \text{ otherwise} \end{cases}$$ 
$$\begin{align*}
  \arg\max_{k'} p(z^i=k'|{x}^i) &=\arg\max_{k'}\frac{\bcancel{\tfrac{1}{K}} N(x^i; {\mu}_{k'}, {I})}{\sum_{j=1}^K \bcancel{\tfrac{1}{K}} N(x^i; {\mu}_j, {I})} \\
  &= \arg\max_{k'} \frac{1}{(2\pi)^{d/2}}\cdot \exp\left (-\frac{1}{2}(x^i-\mu_{k'})^\top(x^i-\mu_{k'})\right )\\
  &= \arg\min_{k'} (x^i-\mu_{k'})^\top(x^i-\mu_{k'}) \\
  &= \arg\min_{k'}\|{x}^i-{\mu}_{k'}\|_2^2
  \end{align*}$$

**update step** follows due to the above hard assignment and the mixture model assumption
  * only update the mean $\mu_k$ based on the assignment
  * as $\pi$ and $\Sigma_k$ are assumed known or fixed

"""

# ╔═╡ 354b5a28-89da-47cf-a792-e6cc13b50ed6
md"""

## Measure clustering performance

When true labels were available 
* need to deals with **labelling** problem
  * K-means/EM might index the labels differently: e.g. 1,2,3 or 3,2,1
* accuracy: percentage of accurately clustered labels
* information based criteria: e.g. **normalised mutual information** (NMI)
  * how much correlation between two clusters 
  * NMI is between 0 and 1; 1 means perfectly correlated, 0 means no correlation at all
* there are others such as random index (RI), adjusted random index (ARI)
* we won't dig into their definitions in details
  * just use them as blackbox for P2

"""

# ╔═╡ 390354c8-1838-461e-b8a5-10d2b218cc51
md"""
## Choosing $K$


We cannot use likelihood to choose $K$, as $K \rightarrow \infty$, the likelihood will increase (out of bound)

* when $K =n$, each observation is one Gaussian with 0 variance
* the likelihood is infinite
* no surprise: likelihood based method favours complicated models, i.e. overfitting

"""

# ╔═╡ a1f3b2e4-2b43-49c6-8da4-06166e15e417
begin
	lls = []
	zs = []
	for k in 1:10
		logLiks, _, _, zs_=em_mix_gaussian(data₂, k)
		push!(zs, zs_)
		push!(lls,logLiks[end])
	end
end

# ╔═╡ 64a3fd07-5d9f-4b46-83e9-dbce03343e19
plot(lls, xlabel="K", ylabel="Log-likelihood", label="")

# ╔═╡ 6f4c6b2a-e20f-4e04-b657-2f82eb5ac4ed
md"""
## Choosing $K$

Like regularisation method, we need to apply some penalties to curb the likelihood

**Bayesian information criteria (BIC)**:

$$\text{BIC}(\mathcal M) = \ln P(D|\theta_{ML}, \mathcal M) -\frac{\text{dim}}{2}\ln n$$

* ``\mathcal M``: model under consideration, e.g. $K=1,2,3\ldots$ for mixture
* the first term is the maximum likelihood achieved
* ``\text{dim}``: the total number of parameters, aka degree of freedom
  * therefore complicated models are penalised
* ``n``: number of training samples





"""

# ╔═╡ b52b8bdf-9000-43c2-8a11-31ce3f7bfe32
function bic_mix_gaussian(logL, K, d, n)
	# dim(π): K; 
	# dim(μ) *K: d*K; 
	# dim(Σ) *K : (d+1)*d/2, symmetric matrix
	dim = K + (d + d*(d+1)/2) * K
	logL - dim/2 * log(n)
end

# ╔═╡ b824a153-80f8-4491-94a1-e4f1554d4e2f
begin
	plot(bic_mix_gaussian.(lls, 1:10, 2, n₂), title="Choose K via BIC", xlabel="K", ylabel="BIC", label="BIC", legend=:bottomright)
	plot!(lls, label="Likelihood")
end

# ╔═╡ aed6328f-b675-4aea-87ee-2f2c83926078
md"""
## *BIC and Bayesian inference

BIC has its root in Bayesian inference
* and *to be completely honest*, all good bits of frequentist machine learning have their roots in Bayesian inference...



In essence, BIC aims at approximating model evidence (which does integration/marginalisation rather than optimisation, therefore avoid overfitting, aka Ocam's Razor)

$$\text{BIC} \approx \ln P(D|\mathcal M)= \ln \int P(D, \theta|\mathcal M)d\theta$$

The approximation is correct due to
* asymptotic behaviour of posterior distribution
  * all posteriors $p(\theta|D)$ asymptotically converge to a Gaussian when enough training data is given (note the integrand is proportional to this posterior)
* then we use *Laplace approximation* to find the approximating Gaussian
* BIC is roughly the normalising constant of the approximating Gaussian
* therefore, it is a terrible metric if you do not have enough data
  * in that case, the posterior is far off from a Gaussian
  * nevertheless, people still use it even it is WRONG, which is sad

"""

# ╔═╡ 3f1a3dc3-20bc-4e5c-8e79-d65c262f0e4e
md"""
## General mixture model

We do not have to make the Gaussian assumption for the mixture components

$p(x)= \sum_{k=1}^K \pi_k \cdot p(x|\phi_k)$

* ``p(x|\phi_k)`` can be any distribution: Poisson, Multinomial, von-Mises Fisher, even linear regresion, logistic regression, neural nets
  * each component has its own parameters (e.g. $\phi_k= \{\mu_k, \Sigma_k\}$ for the Gaussian case)
* all other assumptions are the same, e.g. the data generating process
* in P2, you are going to implement an EM algorithm for mixture of von Mises-Fisher


"""

# ╔═╡ a487fbb6-ed49-4f93-a391-d3b721485709
md"""

## Example of other mixture model


"""

# ╔═╡ 2075d843-45d3-4ef0-9974-c3349913fdad
md"""
For example, the mixture component can be **von Mises-Fisher** (vMF) distribution for directional vectors: ``x\in R^d, \|x\|=1`` 
* vMF is basically Gaussian's counterpart of data on a hypersphere

vMF has a probability density form:

$$p(x|\mu, \kappa) = c_d(\kappa) \exp (\kappa \mu^\top x),$$
where 
* ``\mu \in R^d`` and $\|\mu\|=1$: is the mean direction of the distribution; 
* ``\kappa > 0``: is the concentration parameter
  * larger $\kappa$ more concentrate samples around ``\mu``;
  * small $\kappa$ more spread out
* ``c_d(\kappa)`` is the normalising constant s.t. the probability integrates to one



"""

# ╔═╡ 9eacb581-7fa1-48b6-b414-1df37131b95f
begin
	Kᵥ = 3
	dᵥ = 3
	trueμs = zeros(dᵥ, Kᵥ)
	trueμs[:,1] = [0.5,1, 1]
	trueμs[:,2] = [-1, -1,1]
	trueμs[:,3] = [0,0, 1]
	trueμs = trueμs ./ [norm(trueμs[:,k]) for k in 1:Kᵥ]'
	trueκs = [5.0, 20.0, 80.0]
	vmfs = [VonMisesFisher(trueμs[:,k], trueκs[k]) for k in 1:Kᵥ]
	trueπsᵥ = [0.4, 0.3, 0.3]
	nᵥ = 500
	mixDataᵥ = zeros(dᵥ, nᵥ)
	truezsᵥ = rand(Categorical(trueπsᵥ), nᵥ)
	for i in 1:nᵥ
	    zᵢ = truezsᵥ[i]
	    mixDataᵥ[:,i] = rand(vmfs[zᵢ])
	end
end

# ╔═╡ 7d07a6ee-8b09-483b-aca0-3c5834261ea1
begin
	plotly()
	n = 100
	u = range(0,stop=2*π,length=n);
	v = range(0,stop=π/2,length=n);
	
	x = cos.(u) * sin.(v)';
	y = sin.(u) * sin.(v)';
	z = ones(n) * cos.(v)';
	
	# The rstride and cstride arguments default to 10
	plt=wireframe(x,y,z, color=:gray)
	
	for k = 1:Kᵥ
		plot!([0.0, trueμs[1, k]], [0.0, trueμs[2, k]], [0.0 , trueμs[3, k]], lw=8, c=k, label="μ"*string(k)*"; κ"*string(k)*"="*string(trueκs[k]))
	    scatter!(mixDataᵥ[1,truezsᵥ .==k], mixDataᵥ[2,truezsᵥ .==k], mixDataᵥ[3,truezsᵥ .==k], markersize =2, c=k, label=string("cluster ", k))
	end
	plt
end

# ╔═╡ 4eca5d49-0b96-43fa-baee-67ac1a4ae070
md"""

## *Towards unsupervised learning of mixture

Let's consider **supervised learning** first, i.e. assume we had the labels, **maximum likelihood estimation** aims at optimising

$$\hat \theta = \arg\max_{\theta} P(D|\theta)$$

* observed data: ``D=\{x^i, z^i\}_{i=1}^n``, labels are observed
* model parameters: ``\theta = \{\pi_k, \phi_k\}_{k=1}^K``

The likelihood becomes

$P(D|\theta) = \prod_{i=1}^n p(z^i, x^i) = \prod_{i=1}^n p(z^i)p(x^i|z^i)$

Take log and after some algebra, it can be shown that (check appendix for details)

$$\mathcal L(\theta) = \ln P(D|\theta)=\sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln p(x^i|\phi_k)$$


To optimise $\phi_k$, i.e. find $$\frac{\partial \mathcal L}{\partial \phi_k}$$, we isolate the terms of $\phi_k$ only

$$\mathcal{L}(\phi_k) = \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi_k) + \text{const.}$$

Therefore, the MLE for $\phi_k$ is 

$\hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi)= \arg\max_{\phi} \ln P(D_k|\phi)$

* ``D_k =\{ x^i| z^i=k, \text{ for } i \in 1,\ldots, n\}``
* the MLE of the those data belong to the k-th class!


In summary, for **supervised learning**, **maximum likelihood estimators** for mixture model is 

$$\hat \pi_k = \frac{\sum_{i=1}^n I(z^i= k)}{n}, \hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi)$$

"""

# ╔═╡ 26b09f89-583f-4158-bc34-31566b38805b
md"""

## EM algorithm for general mixture


Similar to EM for Gaussian, the unsupervised learning or **EM algorithm** for general mixture use estimation instead of true labels

Improvise labels in **E step**:

$I(z^i=k) \Rightarrow \underbrace{p(z^i=k|x^i)}_{w_{ik}}$

**M step**:

$$\hat \pi_k \leftarrow \frac{\sum_{i=1}^n w_{ik}}{n}, \hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {w_{ik}} \cdot \ln p(x^i|\phi)$$


* solving a **weighted MLE** for $\phi_k$ instead
* the EM for mixture of Gaussian is actually a specific case, the reestimation for $\mu_k, \Sigma_k$ are the closed form solution of the weighted MLE
"""

# ╔═╡ b3d35154-2917-4252-b70d-7060bc2bfe6e
md"""

## EM algorithm for general mixture

Initilisation: random guess ${\theta} \leftarrow \{{\pi_k}, \phi_k\}_{k=1}^K$


* Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$w_{ik} \leftarrow p(z^i=k|{x}^i) = \frac{\pi_k \cdot p(x^i|\phi_k)}{\sum_{j=1}^K \pi_j \cdot p(x^i| \phi_j)}$$


* Maximisation step (M step): update ${\theta}$, for $k=1\ldots K$

$\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n w_{ik}$

${\phi}_{k} \leftarrow \arg\max_{\phi} \sum_{i=1}^n w_{ik} \ln p(x^i|\phi)$


Repeat above two steps until converge

"""

# ╔═╡ b15297ab-79b0-47f9-adc7-ba1f63ea0735
md"""

* E-step is almost the same for all mixture model (you only need to change the likelihood)
* M-step is **weighted maximum likelihood** estimation
  * some weighted MLE have closed form solution (like Gaussian)
  * if not, we need to apply gradient descent in the M step
    * e.g. mixture of logistic regression 

"""

# ╔═╡ 941c27aa-1f1a-41e9-b086-67d494d16d8f
md"""

## Towards mixture of regression; re-visit linear regression

We need to revisit **linear regression's probabilistic model** first 
* we will show the probabilistic model behind linear regression
* and extend linear regression to mixture

For linear regression, we use **least squared method** to optimise 


$$\text{loss}(\beta) = \sum_{i=1}^n (y^i- \beta^\top x^i)^2$$


Do we have any justification for this loss? Yes, regression's probabilistic model. We assume each target $y^i$ is formulated as a sum of signal (non random) + random noise (Gaussian r.v.; the stochastic bit)

$$y^i = \beta^\top x^i + \epsilon, \;\; \epsilon \sim N(0, \sigma^2)$$

which implies

$p(y^i|x^i, \beta, \sigma^2) = N(y^i; \beta^\top x^i , \sigma^2)= \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left(-\frac{({y}^{i}-{\beta}^\top{{x}}^{i})^2}{2\sigma^2}\right)$

* ``y`` is univariate Gaussian with mean $\beta^\top x$ and variance $\sigma^2$ 
* ``x`` is assumed fixed, i.e. discriminative model


"""

# ╔═╡ af1c07cf-71d5-4ed1-9de0-966d43cd1c71
md"``\beta_1`` $(@bind b_1 Slider(-4:0.5:4, default=3)); ``x_i`` $(@bind xᵢ0 Slider(0:0.1:1, default=0.15));	``\sigma^2`` $(@bind σ²0 Slider(0.01:0.01:2, default=0.5))"

# ╔═╡ 4eba42cc-6383-4a67-adf1-3230fead0824
md"Slope ``\beta_1=``$(b_1); intercept ``\beta_0=-3``"

# ╔═╡ 703b71df-da6c-4614-b888-5b4b09c0fc37
md"input $x^i=$ $(xᵢ0); and ``\sigma^2=`` $(σ²0)"

# ╔═╡ a318ba95-b9f8-4350-9404-11584a2957ae
begin
	p_lr = plot(title="Linear regression's probabilistic model",legend=:bottomright)
	β0 = [-3, b_1]
	n0 = 250
	xx = [ones(n0) rand(n0)]
	yy = xx * β0 + sqrt(σ²0) * randn(n0)
	plot!(xx[: ,2], yy, st=:scatter, label="")
	plot!([0,1], x->β0[1]+β0[2]*x, c= 1, linewidth=5, label="")
	xis = [0, 0.25, 0.5, 0.75, 0.99, xᵢ0]
	for i in 1:length(xis)
		x = xis[i]
		μi = dot(β0, [1, x])
		xs_ = μi-3:0.01:μi+3
		ys_ = pdf.(Normal(μi, sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		if i == length(xis)
			scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label="μ @ x="*string(x), markersize=6)
			plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
		else
			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, label="μ @ x="*string(x))
		end
		
	end
	p_lr	
end

# ╔═╡ 58f87ebb-1ae1-41d1-98df-e2bc478855d3
md"""
## MLE is LSE for regressions

"""

# ╔═╡ 2f540967-79f3-4bdc-a87b-08cd643781f0
md"""

The log transformed likelihood is 

$$\ln p({y}^{i}| {\beta}, \sigma^2, {x}^{i}) = -\frac{1}{2} \ln 2\pi\sigma^2 -\frac{1}{2\sigma^2}({y}^{i}-{\beta}^\top {x}^{i})^2$$


Log-likelihood (the prob of observing the data, i.e. $D=\{y^i\}$, given parameters) is 

$$\begin{align}\mathcal L(\beta, \sigma^2) &= \ln P(D|\beta, \sigma^2) = \ln \prod_{i=1}^n p(y^i|x^i, \beta, \sigma^2)\\
&= \boxed{\sum_{i=1}^n \ln p({y}^{i}| {\beta}, \sigma^2, {x}^{i})} \\
&=  -\frac{n}{2} \ln 2\pi\sigma^2 -\frac{1}{2\sigma^2} \underbrace{\sum_{i=1}^n({y}^{i}-{\beta}^\top {x}^{i})^2}_{\text{sum of squared error loss!}}
\end{align}$$

* we have used independence assumption and logarithm's identity
* maximising the log-likelihood is the same as minimising the squared error
* MLE is the same as least square method
* MLE is better as it has one additional parameter $\sigma^2$ (it becomes more interesting when $\sigma^2$ depends on $x$)!


## *Probabilistic model unifies all ML models

Indeed, probabilistic models unify all (interesting) ML models

$$\begin{align}\mathcal L(\theta) = \ln P(D|\theta) 
&= \boxed{\sum_{i=1}^n \ln p(\text{data}^i| \theta)} 
\end{align}$$

* this formula **applies to all** likelihood based optimisation methods (as long as observation independence is assumed)
  * all loss functions are merely specific cases of this formula
* just plug in different dataset $D$
  * unsupervised learning:  $D=\{x^i\}$ or $\text{data}^i = x^i$ only
  * supervised learning
    * generative model: $D = \{x^i, y^i\}$, or $\text{data}^i = \{x^i, y^i\}$
    * discriminative model: $D = \{y^i\}$, or $\text{data}^i = y^i$
* different likelihood $p$ for your data
  * regression: Gaussian
  * classification: Bernoulli, Multinoulli
* then training is optimising this function

$$\hat \theta \leftarrow \arg\max_{\theta} \mathcal L(\theta)$$

As I said, it applies to all. Regularisation is of course no exception. In essense, we introduce a prior then optimise the posterior instead, therefore they are called maximum a posteriori (MAP) instead of maximum likelihood (ML):

$$p(\beta|y, X, \sigma^2) \propto p(\beta) p(y|X, \beta, \sigma^2)$$

Assume zero mean Gaussian with diagonal covariance prior: $p(\beta)= N(\beta; 0, 1/\lambda\cdot  I)$, which implies

$$\begin{align}\ln p(\beta|y, X, \sigma^2) &= \ln p(\beta)+\ln p(y|X, \beta, \sigma^2) +C \\
&= -\frac{\lambda}{2} \beta^\top\beta - \frac{1}{2 \sigma^2}\sum_{i=1}^n (y^i -\beta^\top x^i)^2+C\end{align}$$ 

Maximising the posterior is the same as minimising the following L2 regularised negative likelihood

$$\frac{1}{2}\sum_{i=1}^n (y^i- \beta^\top x^i)^2 +  \frac{\lambda}{2} \|\beta\|^2$$

L1 regularisation corresponds to assuming a Laplace distribution prior instead of Gaussian.
"""

# ╔═╡ 08361c85-f54e-4aba-aaba-a2b262c4440d
md"""

## Example- mixture of regression (conti.)

What if your data looks like this ?

* one straight line is a bad fit
* quite likely key covariate $z^i\in 1,\ldots,K$ are missing when collecting the data
  * e.g. each observation's gender, or some other categorical feature
  * or due to privacy issue, cannot be collected at all!


We need to resort to the general mixture's algorithm
"""

# ╔═╡ 270de2f3-2549-46f0-95f2-101f415f2646
begin
	nₘₗᵣ = 600
	trueβs = [[-3, 4.0] [4.0, -6.0] [-1, 8.0]]
	trueσ²s = [0.25, 0.25, 1.0]
	X = [ones(nₘₗᵣ) rand(nₘₗᵣ)]
	trueπsₘₗᵣ = [0.4, 0.4, 0.2]	
	truezsₘₗᵣ = rand(Categorical(trueπsₘₗᵣ), nₘₗᵣ)	
	Y = zeros(nₘₗᵣ)
	Kₘₗᵣ = 3
	for i in 1:nₘₗᵣ
		zᵢ = truezsₘₗᵣ[i]
		Y[i] = rand(Normal(X[i,:]' * trueβs[:, zᵢ], sqrt(trueσ²s[zᵢ])))
	end
	dataₘₗᵣ = [X Y]
end;

# ╔═╡ ca781884-6b9c-4fbe-bc92-716cff2b16a7
plot(X[: ,2], Y, st=:scatter, label="", title="Unsupervised learning view")

# ╔═╡ 947544de-3a74-4ecc-b744-14ff9297dc53
md"""
## Example- mixture of regression (conti.)

Mixture of $K$ regression models:

$$p(y^i|{x}^{i}) = \sum_{k=1}^K \pi_k \cdot{p(y^i| {x}^{i}, {\beta}_k,  \sigma_k^2)}=\sum_{k=1}^K \pi_k {N(y^i; {\beta}_k^\top {x}^{i}, \sigma_k^2)}$$

  - ``y^i`` can take one of K possible regression models
  
$$y^i = {\beta}_k^\top {x}^{i} + \epsilon_k, \;\;\; \epsilon_k \sim N(0, \sigma^2_k)\;\;\text{ for }k= 1,\ldots, K$$
  
  - ``\phi_k=\{{\beta}_k, \sigma_k^2 \}`` are the K regression component' parameters
  - ``\pi_k`` again is the popularity of each component apriori
  

"""

# ╔═╡ b99b1177-a0bd-429c-89b0-b1ef141179aa
begin
	p_mlr = plot(title="Mixture of linear regression with true zs")
	for k in 1:Kₘₗᵣ
		plot!(X[truezsₘₗᵣ .== k ,2], Y[truezsₘₗᵣ .==k], st=:scatter, c=k, label="")
		plot!([0,1], x->trueβs[1,k]+trueβs[2,k]*x, c=k, linewidth=4, label="")
	end
	p_mlr
end

# ╔═╡ cca9edfc-37e1-40a0-ba29-21b9b642f133
md"""
## EM algorithm for finite mixture of regression

To solve it, we just apply the general EM algorithm
* E step is the same just sub in the likelihood for $p(y|z, x)$
* M step needs a bit derivation but it solves a weighted likelihood optimisation problem

Initilisation: random guess ${\theta} \leftarrow \{{\pi_k}, \beta_k, \sigma^2_k\}_{k=1}^K$


* Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$w_{ik} \leftarrow p(z^i=k|{x}^i) = \frac{\pi_k \cdot {p(y^i| {x}^{i}, {\beta}_k,  \sigma_k^2)}}{\sum_{j=1}^K \pi_j \cdot {p(y^i| {x}^{i}, {\beta}_j,  \sigma_j^2)}}= \frac{\pi_k \cdot {N(y^i; {\beta}_k^\top {x}^{i},    \sigma_k^2)}}{\sum_{j=1}^K \pi_j \cdot {N(y^i; {\beta}_j^\top {x}^{i},   \sigma_j^2)}}$$


* Maximisation step (M step): update ${\theta}$, for $k=1\ldots K$

$\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n w_{ik}$

${\beta}_{k},\sigma_k^2 \leftarrow \arg\max \sum_{i=1}^n w_{ik} \ln {N(y^i; {\beta}^\top {x}^{i},    \sigma^2)}$


Repeat above two steps until converge


"""

# ╔═╡ fa7e50a6-4797-4327-900b-243614a89072
md"""
## Demonstration of E-step

"""

# ╔═╡ ef636cef-3d51-4cd1-885d-e1b504343838
md"Choose i-th observation: $(@bind iₜₕ Slider(1:length(truezsₘₗᵣ), default=1))"

# ╔═╡ b9f11459-688e-4d5f-925b-8c47b70de6cf
md"``p(z^i|x^i)`` for $(iₜₕ)th observation is"

# ╔═╡ da915f15-1f27-40d5-9cb2-2b746130ebbc
begin
	plotly()
	p_mlr_ = plot(title="E-step of mixture of regression", legend=:topleft)
	for k in 1:3
		plot!(X[truezsₘₗᵣ .== k ,2], Y[truezsₘₗᵣ .==k], st=:scatter, c=k, label="")
		plot!([0,1], x->trueβs[1,k]+trueβs[2,k]*x, c=k, linewidth=4, label="")
	end
	xis_ = [X[iₜₕ,2]]
	for k in 1:Kₘₗᵣ
		x = xis_[1]
		μi = dot(trueβs[:, k], [1, x])
		xs_ = μi-3:0.01:μi+3
		ys_ = pdf.(Normal(μi, sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		scatter!([x],[μi], markerstrokewidth =3, markershape = :diamond, c=k, label="μ"*string(k), markersize=4)
		plot!(ys_ .+x, xs_, c=k, label="", linewidth=2)
	end
	scatter!([X[iₜₕ,2]], [Y[iₜₕ]], markersize = 8, markershape=:xcross, markerstrokewidth=3, c= :white, label="iₜₕ")
	p_mlr_
end

# ╔═╡ c4662e69-a210-4ba3-973b-fc4f95265fec
md"""
## M-step for mixture of regression

We need to solve this **weighted MLE** in the M step

${\beta}_{k},\sigma_k^2 \leftarrow \arg\max \sum_{i=1}^n w_{ik} \ln {N(y^i; {\beta}^\top {x}^{i},    \sigma^2)}$

It turns out there is a closed form analytical solution to this **weighted MLE** (check appendix)

$${\beta}_{k} \leftarrow ({X}^\top {W}_k {X})^{-1} {X}^\top  {W}_k {y}$$
$$\sigma_k^2 \leftarrow \left (\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2\right ) \left(\sum_{i=1}^n w_{ik}\right )^{-1}$$ 

where 

${W}_k = \begin{bmatrix} w_{1k} & 0 & \ldots & 0 \\
0 & w_{2k} & \ldots & 0 \\
\vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & \ldots & w_{mk}
\end{bmatrix}$


It makes sense, if we set all weights to 1  i.e. $w_{ik}=1$,

$W_k=I$ 

then, we **recover** the normal unweighted MLE for linear regression!

$$\beta=({X}^\top {W}_k {X})^{-1} {X}^\top  {W}_k {y}=({X}^\top I {X})^{-1} {X}^\top  I {y} = ({X}^\top {X})^{-1} {X}^\top  {y}$$

$$\begin{align}\sigma^2_k&=\left (\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2\right ) \left(\sum_{i=1}^n w_{ik}\right )^{-1}\\
&= \left (\sum_{i=1}^n  ({y}^{i} - {\beta}^\top {x}^{i})^2\right ) \left(\sum_{i=1}^n 1\right )^{-1} \\
&=\frac{1}{n} \sum_{i=1}^n  ({y}^{i} - {\beta}^\top {x}^{i})^2
\end{align}$$
"""

# ╔═╡ 64798945-3dec-4541-b010-be41de026de5
md"""
## Implementation in Julia

"""

# ╔═╡ 12322892-2c70-47e3-826a-739507b54f1b
md"E-step is almost the same, only differ in the likelihood part."

# ╔═╡ e554f436-c07d-4450-8a5a-9257d8cfb03b
function e_step_mix_reg(data, βs, σ²s, πs)
	X = data[:, 1:end-1]
	y = data[:, end]
	K = length(πs)
	# this is the only line that is different from EM mixture Gaussians
	logLiks = hcat([logpdf.(Normal.(X * βs[:,k], sqrt(σ²s[k])), y) for k in 1:K]...)
	logPost = log.(πs') .+ logLiks
	logsums = logsumexp(logPost, dims=2)
	ws = exp.(logPost .- logsums)
	return ws, sum(logsums)
end

# ╔═╡ 33fde9a8-d3dd-4dd6-9f22-0345afde275c
wsₘₗᵣ,_ = e_step_mix_reg(dataₘₗᵣ, trueβs, trueσ²s, trueπsₘₗᵣ) ;

# ╔═╡ fd466a8b-eef3-4900-a0c2-f98a50c1a666
wsₘₗᵣ[iₜₕ,:]

# ╔═╡ 572b0598-0ae2-4863-9d0b-dd702bb237c6
md"Basically, direct translation of the weighted MLE to code"

# ╔═╡ d9d8a0a9-1cb2-4d3b-ad36-01d58ff96752
function m_step_mix_reg(data, ws)
	n, K = size(ws)
	X = data[:, 1:end-1]
	_, d = size(X)
	y = data[:, end]
	βs = zeros(d,K)
	σ²s = ones(K)
	ns = sum(ws, dims=1)
	πs = ns ./sum(ns)
	for k in 1:K
		Wₖ = Diagonal(ws[:,k])
		βs[:, k] = βₖ =(X'* Wₖ * X)^(-1) * X' * Wₖ * y
		σ²s[k] = sum(ws[:,k] .* (y - X * βₖ).^2) / ns[k]
	end
	return πs[:], βs, σ²s
end

# ╔═╡ 8cd6f35a-ccce-40e4-a418-bc647a5929ea
md"It is left as an exercise to write an EM algorithm by using the E and M step methods. It should be bloodily easy: just a simple loop plus convergence check."

# ╔═╡ 7b24424e-e7f2-452d-96e5-af2c69e49169
md"""

## Demonstration
"""

# ╔═╡ f062e3a4-69ae-462e-9ee6-54c7f2e5611f
begin
	gr()
	plEMₘₗᵣ = []
	zs0 = rand(1:Kₘₗᵣ, size(dataₘₗᵣ)[1])
	ws0 = Matrix(I, Kₘₗᵣ, Kₘₗᵣ)[zs0,:]
	l_ = Inf
	anim = @animate for iter in 1:30
		πs0, βs0, σ²s0 = m_step_mix_reg(dataₘₗᵣ, ws0)
		p = plot(title="Iteration: "*string(iter)*" L="*string(round(l_, digits=2)))
		for k in 1:Kₘₗᵣ
			plot!(X[zs0 .== k ,2], Y[zs0 .==k], st=:scatter, c=k, label="")
			plot!([0,1], x->βs0[1,k]+βs0[2,k]*x, c=k, linewidth=4, label="")
		end
		
		ws0, l_ = e_step_mix_reg(dataₘₗᵣ, βs0, σ²s0, πs0)
		zs0 = argmax.(eachrow(ws0))
		push!(plEMₘₗᵣ, p)
	end
end

# ╔═╡ 35246910-56ad-4da5-b9bb-b6803e4ccb52
gif(anim, fps=5)

# ╔═╡ 10a70425-d164-4c48-9731-5eeef44a30f1
md"""

## Mixture of vMFs

EM for mixture of vMFs behaves very similar, check the gif below (you need to implement a learning algorithm for this model in P2)
"""

# ╔═╡ df214b12-5bea-4496-9070-65fc406aad4e
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5014/vmfmix2.gif' width = '500' /></center>"

# ╔═╡ 7f94600f-d7f6-4923-a6b8-456c2ced85f1
md"""

## Suggested reading

Machine learning: a probabilistic approach by Kevin Murphy
* 11.2 and 11.4: mixture models
* Chapter 11: general EM algorithm


"""


# ╔═╡ daf68d56-df9d-4d91-9952-7c917aefcc37
md"""
## Appendix
"""

# ╔═╡ 8a26a68f-51c6-4138-818c-260b616518c9
md"""
## *Cosine similarity metric

Similar to Gaussian, the kernel $\kappa \mu^\top x$ measures *similarity* between ``\mu`` and ``x``

$$\mu^\top x = \|\mu\| \cdot \|x\|\cdot  \text{cos}(\theta)= \text{cos}(\theta)$$

* for unit vectors, inner product relates to the angle ``\theta`` between $x$ and $\mu$
* ``\text{cos}(\theta) \in [-1, 1]``
  * when $x, \mu$ point to the same direction, $\text{cos}(\theta)=1$ reach its maximum, so is $p(x)$
  * when $x, \mu$ point to the opposite direction, $\text{cos}(\theta)=-1$ has its minimum, so is $p(x)$
"""

# ╔═╡ 4d98a224-f199-462b-b526-b952f7c266ab
md"Demo of cosine similarity distance metric"

# ╔═╡ 56e56d06-e24f-42d8-b64a-58b890c4f749
begin
	# polar coordinate
	xₜ(t) = cos(t)
	yₜ(t) = sin(t)
	μ0 = [1,1]/norm([1,1])
	κ0 = 1.0
	θ0 = acos(dot(μ0,[1,0]))
	vmf = VonMisesFisher(μ0, κ0)
	animᵥ = @animate for θ in θ0:0.25:θ0+2π
		pltᵥ=plot(xₜ, yₜ, 0, 2π, leg=false, linewidth=2, ratio=1, xlim=[-1, 1])
		quiver!([0], [0], quiver=([μ0[1]], [μ0[2]]), linewidth=2, color=:red)
		x0 = [cos(θ), sin(θ)]
		quiver!([0], [0], quiver=([x0[1]], [x0[2]]), linewidth=1, color=:black)
		title!("θ="*string(round(θ-θ0,digits=2))*"; μᵀx="*string(round(cos(θ-θ0), digits=2))*"; p(x)="*string(round(pdf(vmf, x0), digits=2)) )
	end
end;

# ╔═╡ 63623430-bf7f-41a3-886f-3e80278e3de0
gif(animᵥ, fps=1.5)

# ╔═╡ 6fe234a6-691b-430e-8ba1-44af78100137
md"""
## *Expectation

Expectation is defined as 

$E[f(X)] = \sum_{x} f(x) \cdot p(X=x)$

Basically it is a weighted average


Example 1, find the expectation of a Bernoulli r.v. $X$ with probability $p$:

$$E[X] = 1 \cdot p(X=1) + 0 \cdot p(X=0) = 1\cdot p + 0 \cdot (1-p) = p$$

* ``f`` is a identity function

Example 2, assume $p(z=k) = w_k$, i.e. $z$ is a multinoulli r.v. then

$E[I(z=k)] = \sum_{k'=1}^K I(z=k) \cdot p(z=k') = w_k$

* the expected value of $I(z=k)$ is just $w_k$
* ``f(z) \triangleq I(z=k)``
"""

# ╔═╡ 3eedbae2-faf5-4cd4-86a2-3386152e17f1
md"""
## *Derivation of MLE for supervised learning of general mixture

To take derivatives, we need to write down the distribution with $I(\cdot)$ notation 

``z^i`` is a multinoulli random variable (like throwing a ``K`` facet die)

$$p(z^i) = \begin{cases}\pi_1 & z^i=1 \\ \pi_2 & z^i=2\\ \vdots & \vdots \\ \pi_K & z^i=K\end{cases}$$

which can be compactly written as 

$$p(z^i) = \prod_{k=1}^K \pi_k^{I(z^i=k)}$$ and also the likelihood model for $x$

$$p(x^i|z^i) = \prod_{k=1}^K p(x^i|\phi_k)^{I(z^i=k)}$$

Their logs are 

$$\ln p(z^i) = \sum_{k=1}^K {I(z^i=k)} \cdot \ln \pi_k;\;\; \ln p(x^i|z^i) =  \sum_{k=1}^K {I(z^i=k)} \cdot \ln p(x^i|\phi_k)$$

Then the log-likelihood becomes

$$\mathcal L(\theta) = \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln p(x^i|\phi_k)$$

To optimise $\phi_k$, we can isolate the terms and write $\mathcal L$ as a function of $\phi_k$ only:

$\mathcal L(\phi_k) = \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi_k) +C$

which provides us the MLE for $\phi_k$

* the pooled MLE for the k-th class's observations!



"""

# ╔═╡ f6c86042-56a7-431c-97f8-28277a880474
md"""
## Weighted MLE for linear regression


The weighted log likelihood function can be written as 

$$\begin{align}
\mathcal{L}({\beta}, \sigma^2) &= \sum_{i=1}^n w_{ik} \log N(y^i; {\beta}^\top {x}, \sigma^2) \\
&= \sum_{i=1}^n   w_{ik} \left(\frac{1}{2} \ln 2\pi\sigma^2 +\frac{1}{2\sigma^2}({y}^{i}- {\beta}^\top {x}^{i})^2 \right )\\
&= \frac{1}{2} \sum_{i=1}^n w_{ik}\ln 2\pi\sigma^2 + \frac{1}{2\sigma^2}\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2 
\end{align}$$


where we have used the Gaussian's density function. Therefore, maximising the weighted log likelihood w.r.t ${\beta}$ amounts to minimising the following weighted least squares:

$$L({\beta}) = - \frac{1}{2\sigma^2}\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2 +C,$$ 

where $C$ denotes all other terms considered as constant when we take the derivative w.r.t ${\beta}$. Note the added negative sign (we are minimising the negative log likelihood). It is easier to derive the estimator by using matrix notation. The above weighted least square can be written as 


$$L({\beta}) = - \frac{1}{2\sigma^2}  ({y} - {X}{\beta})^\top {W}_k ({y} - {X}{\beta}) +C,$$ 

where 

${W}_k = \begin{bmatrix} w_{1k} & 0 & \ldots & 0 \\
0 & w_{2k} & \ldots & 0 \\
\vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & \ldots & w_{nk}
\end{bmatrix}$ is a diagonal matrix with $w_{ik}$ for $i=1,\ldots, n$ as the diagonal entries. 


Take the derivative with respect to ${\beta}$ we have

$$\begin{align}\frac{\partial L({\beta})}{\partial {\beta}} = - \frac{1}{2\sigma^2} 2({y}- {X\beta})^\top {W}_k (-{X}) 
\end{align}$$ 

and set it to zero 

$$\begin{align} &- \frac{1}{2\sigma^2}2\cdot ({y}- {X\beta})^\top {W}_k (-{X})  =0\\
&\Rightarrow {X}^\top {W}_k {X} {\beta} = {X}^\top {W}_k {y} \\
&\Rightarrow {\beta} = ({X}^\top {W}_k {X})^{-1} {X}^\top {W}_k {y} ,
\end{align}$$

Note that if you set ${W}_k= {I}$, the ordinary least square estimator is recovered.
"""

# ╔═╡ de4f20a9-81d1-4004-90c6-f61e873ee23c
md"""
For $\sigma_k^2$, take derivative

$$\begin{align*}
\frac{\partial L(\sigma^2)}{\partial \sigma^2} &= \left (-\frac{1}{2} \sum_{i=1}^n w_{ik}\right )\frac{1}{(2\pi\sigma^2)}2\pi - \left (\frac{1}{2}\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2\right )(-1)(\sigma^2)^{-2} \\
& = \left (-\frac{1}{2} \sum_{i=1}^n w_{ik}\right )\frac{1}{\sigma^2}+ \left (\frac{1}{2}\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2\right )(\sigma^2)^{-2}
\end{align*}$$

Set it to zero we have 

$$\sigma^2 = \left (\sum_{i=1}^n w_{ik} ({y}^{i} - {\beta}^\top {x}^{i})^2\right ) \left(\sum_{i=1}^n w_{ik}\right )^{-1}.$$ Again if you set $w_{ik}=1$ for all $i=1,\ldots, n$, the estimator is the same as the ordinary linear regression.
"""

# ╔═╡ 7bd9f612-c78e-44dd-9426-0c09e81f3800
md"## Code used for this lecture"

# ╔═╡ bc8c1142-c9d7-4a3e-94b1-e4330139ed1f
qdform(x, S) = dot(x, S, x)

# ╔═╡ d3bb7184-dc1a-4c79-93f7-354b2a70363b
# decision boundary function of input [x,y] 
function decisionBdry(x,y, mvns, πs)
	z, _ = e_step([x,y]', mvns, πs)
	findmax(z[:])
end

# ╔═╡ 505aa001-cf2c-4f43-b215-c718bfaec9f3
function logLikMixGuassian(x, mvns, πs, logLik=true) 
	l = logsumexp(log.(πs) .+ [logpdf(mvn, x) for mvn in mvns])
	logLik ? l : exp(l)
end

# ╔═╡ caaf594e-bb3a-42cf-9a73-646bc5b9a665
function plot_clusters(D, zs, K, loss=nothing, iter=nothing)
	title_string = ""
	if !isnothing(iter)
		title_string ="Iteration: "*string(iter)*";"
	end
	if !isnothing(loss)
		title_string *= " L = "*string(round(loss; digits=2))
	end
	plt = plot(title=title_string, ratio=1)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label="cluster "*string(k))
	end
	return plt
end

# ╔═╡ aba4dfb6-0ba1-43d6-ba67-55b7a35caeb4
begin
	plt₁₀ = plot_clusters(data₂, argmax.(eachrow(zs[10])), 10)
	title!(plt₁₀, "K=10")
end

# ╔═╡ d1a027b8-61f6-41be-8b3a-2da864195d2c
# plot type: cl: classification; db: decision boundary; ct: contour
function mixGaussiansDemoGif(data, K, iters = 10; init_step="e", add_contour=false)
	# only support 2-d
	dim = 2 
	anims = [Animation() for i in 1:3]
	if init_step == "e"
		zs_ = rand(1:K, size(data)[1])
		zs = Matrix(I,K,K)[zs_,:]
		l = Inf
	else
		ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		ms .+= randn(dim,K)
		mvns = [MvNormal(ms[:,k], Matrix(1.0I,dim,dim)) for k in 1:K]
		zs, l = e_step(data, mvns, 1/K .* ones(K))
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	xs = (minimum(data[:,1])-0.1):0.1:(maximum(data[:,1])+0.1)
	ys = (minimum(data[:,2])-0.1):0.1:(maximum(data[:,2])+0.1)
	cs = cgrad(:lighttest, K+1, categorical = true)

	for iter in 1:iters
		# M step
		mvns, ps  = m_step(data, zs)
		# animation 1: classification evolution 
		p1 = plot_clusters(data, zs_, K, l, iter)
		if add_contour
			for k in 1:K 
				plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
				scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			end
		end
		frame(anims[1], p1)
		# animation 2: decision boundary evolution 
		# p2 = contour(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2], nlev=K, fill=true, c=cgrad(:lighttest, K+1, categorical = true), leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1)
		# for k in 1:K
		# 	scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= cs[k])
		# end
		# frame(anims[2], p2)

		# animation 3: contour plot
		# p3 = plot_clusters(data, zs_, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1)
		# for k in 1:K
		# 	scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= cs[k], label="")
		# end
		frame(anims[3], p3)
		# E step
		zs, l = e_step(data, mvns, ps)
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	return anims
end

# ╔═╡ 31cb6bdb-a472-488e-83d4-3eb6c6a2fe82
begin
	function sampleMixGaussian(n, mvns, πs)
		d = size(mvns[1].Σ)[1]
		samples = zeros(n, d)
		# sample from the multinoulli distribution of cⁱ
		cs = rand(Categorical(πs), n)
		for i in 1:n
			samples[i,:] = rand(mvns[cs[i]])
		end
		return samples, cs
	end
end

# ╔═╡ 63fea13c-6b1f-44ee-b8b9-3486127e3714
md"Datasets"

# ╔═╡ dd3ca487-a920-4cf8-9406-deca4979231b
begin
	K₃ = 3
	trueπs₃ = [0.25, 0.5, 0.25]
	trueμs₃ = [[1, 1] [0.0, 0] [-1, -1]]
	trueΣs₃ = zeros(2,2,K₃)
	trueΣs₃ .= [1 -0.9; -0.9 1]
	trueΣs₃[:,:,2] = [1 0.9; 0.9 1]
	truemvns₃ = [MvNormal(trueμs₃[:,k], trueΣs₃[:,:,k]) for k in 1:K₃]
	n₃ = 200* K₃
	data₃, truezs₃ = sampleMixGaussian(200, truemvns₃, trueπs₃)
	data₃test, truezs₃test = sampleMixGaussian(100, truemvns₃, trueπs₃)
	xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
end;

# ╔═╡ ffb397ae-2a6c-4ae3-a36e-7bb6c16b0f74
begin
	gr()
	pltqda₂ = plot(title="Mixture model of Gaussian with true zs", ratio=1)
	xs_qda₃ = minimum(data₃[:,1])-0.1:0.1:maximum(data₃[:,1])+0.1
	ys_qda₃ = minimum(data₃[:,2])-0.2:0.1:maximum(data₃[:,2])+0.2
	for k in 1:K₃
		scatter!(data₃[truezs₃ .==k,1], data₃[truezs₃ .==k, 2], label="", c= k)
		scatter!([truemvns₃[k].μ[1]], [truemvns₃[k].μ[2]], color = k, label = "μ"*string(k), markersize = 10, markershape=:diamond, markerstrokewidth=3)
		contour!(xs_qda₃, ys_qda₃, (x,y)-> pdf(truemvns₃[k], [x,y]), levels=3, colorbar = false, ratio=1,lw=5) 
	end

	pltqda₂
end

# ╔═╡ 7c504c9d-69b3-4d3c-860c-581feae73f67
begin
	plotly()
	logPx = false
	xs = (minimum(data₂[:,1])):0.1: (maximum(data₂[:,1]))
	ys = (minimum(data₂[:,2])):0.1: (maximum(data₂[:,2]))
	plt_mix_contour =plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns₃, πs0, logPx), st=:contour,fill = true, ratio=1, colorbar=false, title="contour plot p(x)")
	plt_mix_surface=plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns₃, πs0, logPx), st=:surface, fill = true, color =:lighttest, ratio=1,colorbar=false, title="density plot p(x)")
	plot(plt_mix_contour, plt_mix_surface)
end

# ╔═╡ 9b147d9a-8a13-478f-b2ef-75fb77ea2913
begin
	gr()
	mixAnims₃ = mixGaussiansDemoGif(data₃, K₃, 100; init_step="e", add_contour=true)
	mixAnims₃_ = mixGaussiansDemoGif(data₃, K₃, 100; init_step="m", add_contour=true)
end

# ╔═╡ 569c3285-4a99-424f-870a-e17cc59ea94b
gif(mixAnims₃[1], fps=20)

# ╔═╡ 21478a69-4b95-4ebd-ab80-5a2cd1caf1f6
gif(mixAnims₃_[1], fps=20)

# ╔═╡ 5bfcaed6-8e02-4e2b-b095-b0a1148eb23b
begin
	# best possible estimate of cluster labels
	zs_ub₃, _=e_step(data₃, truemvns₃, trueπs₃);
	zs_ub₂, _=e_step(data₂, truemvns₂, trueπs₂);
	# performance upper bound
	nmi_ub_d₃ = mutualinfo(argmax.(eachrow(zs_ub₃)), truezs₃)
	nmi_ub_d₂ = mutualinfo(argmax.(eachrow(zs_ub₂)), truezs₂)
end;

# ╔═╡ afcdfadb-d2f4-4214-9134-f6ccad333074
begin
	function assignment_step(D, μs)
		_, K = size(μs)
		distances = hcat([sum((D .- μs[:,k]').^2, dims=2) for k in 1:K]...)
		min_dis, zs_ = findmin(distances, dims=2)
		# zs_ is a cartesian tuple; retrieve the min k for each obs.
		zs = [c[2] for c in zs_][:]
		return min_dis[:], zs
	end

	function update_step(D, zs, K)
		_,d = size(D)
		μs = zeros(d,K)
		# update
		for k in 1:K
			μₖ = mean(D[zs.==k,:], dims=1)[:]
			μs[:,k] = μₖ
		end
		return μs
	end
end

# ╔═╡ 4b5bd5e9-297b-4e2b-9732-4110692b79fd
function kmeans(D, K=3; tol= 1e-4, maxIters= 100)
	# initialise
	n, d = size(D)
	zs = rand(1:K, n)
	μs = D[rand(1:n, K),:]'
	loss = zeros(maxIters)
	i = 1
	while i <= maxIters
		# assigment
		min_dis, zs = assignment_step(D, μs)
		# update
		μs = update_step(D, zs, K)
		
		loss[i] = sum(min_dis)

		if i > 1 && abs(loss[i]-loss[i-1]) < tol
			i = i + 1
			break;
		end
		i = i + 1
	end
	return loss[1:i-1], zs, μs
end

# ╔═╡ a8d32551-a88b-4da4-ae5a-e98f888937a2
begin
	_, zskm₂, _ = kmeans(data₂, K₂)
	_, zskm₃, _ = kmeans(data₃, K₃)
	ll₃, gss₃, pps₃, zs₃=em_mix_gaussian(data₃, K₃; init_step="m")
	ll, gss₂, pps₂, zs₂=em_mix_gaussian(data₂, K₂; init_step="m");
end;

# ╔═╡ c9dfa737-02d8-43b9-9ee8-d5ef26c4b1fb
begin
	nmi_em_d₃ = mutualinfo(argmax.(eachrow(zs₃)), truezs₃)
	nmi_em_d₂ = mutualinfo(argmax.(eachrow(zs₂)), truezs₂)
	nmi_km_d₂ = mutualinfo(zskm₂, truezs₂)
	nmi_km_d₃ = mutualinfo(zskm₃, truezs₃)
end;

# ╔═╡ 2118e8c6-ee65-4ec4-9b20-81cf094e72c3

md"""
The NMI for the two challenging datasets  (you need to repeat a few times to avoid local optimum)

||dataset 2| dataset 3|
|:---:|:---:|:---:|
|Kmeans| $(round(nmi_km_d₂, digits=2))| $(round(nmi_km_d₃, digits=2))|
|EM |$(round(nmi_em_d₂, digits=2))| $(round(nmi_em_d₃, digits=2)) |

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Clustering = "~0.14.2"
Distributions = "~0.25.49"
Plots = "~1.25.12"
PlutoUI = "~0.7.35"
StatsBase = "~0.33.16"
StatsFuns = "~0.9.16"
StatsPlots = "~0.14.33"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "2dd813e5f2f7eec2d1268c57cf2373d3ee91fcea"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.1"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0597dffe1268516192ff4ddebdb4d8937254512d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.63"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9a36165cf84cff35851809a40a928e1103702013"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ca433b9e2f5ca3a0ce6702a032fce95a3b6e1e48"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.14"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d16070abde61120e01b4f30f6f398496582301d6"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.12"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "48598584bacbebf7d30e20880438ed1d24b7c7d6"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.18"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "4d9c69d65f1b270ad092de0abe13e859b8c55cad"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.33"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═d5f4454a-997f-11ec-215d-2f93f237aa1b
# ╠═566619ff-f7cd-44c3-8620-58544fb8492f
# ╟─3a68f4fd-c333-4293-b1b4-bff8eff99409
# ╟─2437de37-77ad-453c-b974-9a31549938b7
# ╟─59a76268-f094-4504-bf7c-046f651f956b
# ╟─56785ca6-3af0-460f-9782-2a1eccaeb1c0
# ╟─817128a7-a4e6-4616-bd95-33459fa25866
# ╟─c1f19943-ea5b-4314-b775-841ef6da6d29
# ╟─d96e3952-0d74-4eaf-b614-40ae0f19aa33
# ╟─6f062399-da4e-4a3b-a500-68a6d618a15f
# ╟─31828fb6-4878-44e9-bef8-7b28e910b4f6
# ╟─a918e796-93a7-4f40-aaaf-3e3a80855430
# ╟─beae3af1-db1f-4a5b-b23d-f6b565028f61
# ╟─5950b105-6264-4837-8f5f-8abe78638b59
# ╟─3d9a9d05-a9ae-4eaa-bd5d-da108ee9c903
# ╟─729480eb-18d1-403c-acf8-d4d85882231f
# ╟─65c8c34a-575d-4abc-9db7-547baff43897
# ╟─ffb397ae-2a6c-4ae3-a36e-7bb6c16b0f74
# ╟─67b0a6ec-0db3-4e1c-a187-147287749c28
# ╟─9c2c18f4-22ab-4c75-bcc2-7a649778ed0f
# ╟─ee03f8bf-042a-4f6a-b64a-12e17013978e
# ╟─5011c678-e36d-4fc0-9705-0847823eee7a
# ╟─7c504c9d-69b3-4d3c-860c-581feae73f67
# ╟─8b8f7e0b-3c56-4c89-aa72-21465b993ecf
# ╠═c235cffa-9612-45b7-aee8-9d4747f761de
# ╠═fa05117f-f9c3-4735-a45e-5bff4cbc7447
# ╠═4dfbd0d2-1ca2-4495-b387-f1f0aaa9a85c
# ╟─17445efc-3733-44ed-bc7a-6e2a36efa63b
# ╟─3be5f4f5-9e1e-46b0-8b34-551d7438ab0d
# ╟─9b147d9a-8a13-478f-b2ef-75fb77ea2913
# ╟─569c3285-4a99-424f-870a-e17cc59ea94b
# ╟─21478a69-4b95-4ebd-ab80-5a2cd1caf1f6
# ╟─983e0e36-cadc-428a-8eb2-06d710fc300a
# ╟─354b5a28-89da-47cf-a792-e6cc13b50ed6
# ╟─2118e8c6-ee65-4ec4-9b20-81cf094e72c3
# ╠═a8d32551-a88b-4da4-ae5a-e98f888937a2
# ╠═c9dfa737-02d8-43b9-9ee8-d5ef26c4b1fb
# ╟─5bfcaed6-8e02-4e2b-b095-b0a1148eb23b
# ╟─390354c8-1838-461e-b8a5-10d2b218cc51
# ╟─a1f3b2e4-2b43-49c6-8da4-06166e15e417
# ╟─64a3fd07-5d9f-4b46-83e9-dbce03343e19
# ╟─aba4dfb6-0ba1-43d6-ba67-55b7a35caeb4
# ╟─6f4c6b2a-e20f-4e04-b657-2f82eb5ac4ed
# ╟─b52b8bdf-9000-43c2-8a11-31ce3f7bfe32
# ╟─b824a153-80f8-4491-94a1-e4f1554d4e2f
# ╟─aed6328f-b675-4aea-87ee-2f2c83926078
# ╟─3f1a3dc3-20bc-4e5c-8e79-d65c262f0e4e
# ╟─a487fbb6-ed49-4f93-a391-d3b721485709
# ╟─2075d843-45d3-4ef0-9974-c3349913fdad
# ╟─7d07a6ee-8b09-483b-aca0-3c5834261ea1
# ╟─9eacb581-7fa1-48b6-b414-1df37131b95f
# ╟─4eca5d49-0b96-43fa-baee-67ac1a4ae070
# ╟─26b09f89-583f-4158-bc34-31566b38805b
# ╟─b3d35154-2917-4252-b70d-7060bc2bfe6e
# ╟─b15297ab-79b0-47f9-adc7-ba1f63ea0735
# ╟─941c27aa-1f1a-41e9-b086-67d494d16d8f
# ╟─4eba42cc-6383-4a67-adf1-3230fead0824
# ╠═703b71df-da6c-4614-b888-5b4b09c0fc37
# ╟─af1c07cf-71d5-4ed1-9de0-966d43cd1c71
# ╠═a318ba95-b9f8-4350-9404-11584a2957ae
# ╟─58f87ebb-1ae1-41d1-98df-e2bc478855d3
# ╟─2f540967-79f3-4bdc-a87b-08cd643781f0
# ╟─08361c85-f54e-4aba-aaba-a2b262c4440d
# ╟─270de2f3-2549-46f0-95f2-101f415f2646
# ╠═ca781884-6b9c-4fbe-bc92-716cff2b16a7
# ╟─947544de-3a74-4ecc-b744-14ff9297dc53
# ╟─b99b1177-a0bd-429c-89b0-b1ef141179aa
# ╟─cca9edfc-37e1-40a0-ba29-21b9b642f133
# ╟─fa7e50a6-4797-4327-900b-243614a89072
# ╟─b9f11459-688e-4d5f-925b-8c47b70de6cf
# ╟─fd466a8b-eef3-4900-a0c2-f98a50c1a666
# ╟─ef636cef-3d51-4cd1-885d-e1b504343838
# ╟─33fde9a8-d3dd-4dd6-9f22-0345afde275c
# ╟─da915f15-1f27-40d5-9cb2-2b746130ebbc
# ╟─c4662e69-a210-4ba3-973b-fc4f95265fec
# ╟─64798945-3dec-4541-b010-be41de026de5
# ╟─12322892-2c70-47e3-826a-739507b54f1b
# ╠═e554f436-c07d-4450-8a5a-9257d8cfb03b
# ╟─572b0598-0ae2-4863-9d0b-dd702bb237c6
# ╠═d9d8a0a9-1cb2-4d3b-ad36-01d58ff96752
# ╟─8cd6f35a-ccce-40e4-a418-bc647a5929ea
# ╟─7b24424e-e7f2-452d-96e5-af2c69e49169
# ╠═f062e3a4-69ae-462e-9ee6-54c7f2e5611f
# ╠═35246910-56ad-4da5-b9bb-b6803e4ccb52
# ╟─10a70425-d164-4c48-9731-5eeef44a30f1
# ╟─df214b12-5bea-4496-9070-65fc406aad4e
# ╟─7f94600f-d7f6-4923-a6b8-456c2ced85f1
# ╟─daf68d56-df9d-4d91-9952-7c917aefcc37
# ╟─8a26a68f-51c6-4138-818c-260b616518c9
# ╟─4d98a224-f199-462b-b526-b952f7c266ab
# ╟─56e56d06-e24f-42d8-b64a-58b890c4f749
# ╟─63623430-bf7f-41a3-886f-3e80278e3de0
# ╟─6fe234a6-691b-430e-8ba1-44af78100137
# ╟─3eedbae2-faf5-4cd4-86a2-3386152e17f1
# ╟─f6c86042-56a7-431c-97f8-28277a880474
# ╟─de4f20a9-81d1-4004-90c6-f61e873ee23c
# ╟─7bd9f612-c78e-44dd-9426-0c09e81f3800
# ╠═bc8c1142-c9d7-4a3e-94b1-e4330139ed1f
# ╠═d3bb7184-dc1a-4c79-93f7-354b2a70363b
# ╠═505aa001-cf2c-4f43-b215-c718bfaec9f3
# ╠═d1a027b8-61f6-41be-8b3a-2da864195d2c
# ╠═caaf594e-bb3a-42cf-9a73-646bc5b9a665
# ╠═31cb6bdb-a472-488e-83d4-3eb6c6a2fe82
# ╟─63fea13c-6b1f-44ee-b8b9-3486127e3714
# ╠═dd3ca487-a920-4cf8-9406-deca4979231b
# ╠═afcdfadb-d2f4-4214-9134-f6ccad333074
# ╠═4b5bd5e9-297b-4e2b-9732-4110692b79fd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
