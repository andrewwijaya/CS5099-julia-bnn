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

# ╔═╡ f5a442b0-eb0c-11ec-183c-556140c4a65e
begin
	using Flux
	using Zygote, Distributions, DistributionsAD, LinearAlgebra
	using StatsFuns
	using Random
	using FiniteDifferences
	using ForwardDiff
	# using PyPlot
	using Plots
	using StatsPlots
	using PlutoUI
	# using PlotlyBase
	# import PlotlyJS:savefig
	# using PlotlyJS
	using LaTeXStrings
	plotly()
end

# ╔═╡ 10cfeab1-5115-4473-98be-a1e678432762
md"""

## Part 1. Basis expansion and regularisation

*Fixed basis expansion* (read Chapter 7.2 of MLAPP or Bishop's chapter 3) can be viewed as a Neural Net's hidden layer. The difference is: the hidden layer does not change or assumed known for the fixed basis expansion model. In other words, when you apply gradient descent for the neural net, the gradient back-propogation stops at the second last layer. And there is no fundamental difference between these two. Also check [Lecture 18 Neural Networks](https://studres.cs.st-andrews.ac.uk/2020_2021/CS5014/Lectures/CS5014_L18_Neural_Network.pdf) to see the connection.

1. Consider the following regression dataset with $N=50$ observations, and `xsamples` as input and `ts` as targets. Apply Radial basis function (RBF) to expand the design matrix from $R^1$ to $R^{50}:$ $$\text{rbf}(x; \mu, \sigma) = \text{exp}\left (- \frac{(x-\mu)^2}{2\sigma^2}\right ),$$
   * choose the expansion locations as `xsamples`
   * and fix $\sigma^2 = 0.5$

2. Fit the regression by using MLE (do not forget the intercept term) and 
3. Add a zero mean Gaussian prior (assume the variance is say 1.0 but implement it as an input parameter), find the posterior distribution
   * also find the MAP or ridge regression
4. Compare MLE and Bayesian's prediction; which one is better ?
5. Now optimise the prior distribution's variance (or precision) by evidence procedure (Chapter 7.6.4 of MLAPP). And compare with previous methods. 
"""

# ╔═╡ 1933425b-3d88-44d1-9cda-a600b5863d1e
begin
	# generate the hidden signal 
	function signal_(x, k= -0.2)
		if (x> -1) && (x < 1) 
			return 1
		elseif (x>-8) && (x<-7)
			return k * x + (1.6- k*(-8))
		else
			return 0
		end
	end
	N = 50
	σ² = 4*1e-2
	# xsamples = rand(N) * 20 .- 10 
	xsamples_ = collect(range(-10., 10., length=N))
	xsamples = shuffle(xsamples_)[1:N]
	# generate targets ts or y
	ts = signal_.(xsamples) + rand(Normal(0, sqrt(σ²)), N)
end

# ╔═╡ 52352f36-9ff3-4f60-afca-880c3a7b1214
md"""

**1). Basis expansion**

"""

# ╔═╡ 984f132a-9edc-4a15-a3da-811c73d032b5
begin
	# Radial basis function
	# called Gaussian kernel in the paper
	function rbf_basis_(x, μ, σ²)
		exp.(-1 ./(2 .* σ²) .* (x .- μ).^2)
	end

	# vectorised version of basis expansion by RBF
	function basis_expansion(xtest, xtrain, σ²=1, intercept = false)
		# number of rows of xtrain, which is the size of basis
		#n_basis = size(xtrain)[1] 
		n_obs = size(xtest)[1]
		X = rbf_basis_(xtest, xtrain', σ²)
		intercept ? [ones(n_obs) X] : X
	end

end

# ╔═╡ 0a72dd30-b0e8-4750-a6e9-add45e7f8f36
begin
	xs = -10:0.1:10
	plot(xs, rbf_basis_(xs, xsamples[1:end]', 1), label="")
end

# ╔═╡ 330b8041-c4c7-43ca-abde-3d65107c5377
xsamples

# ╔═╡ 0ea6f598-7d8b-4d4a-95a9-46abfdd655ee
begin
	intercept = true
	σϕ²= 0.5
	Φ = basis_expansion(xsamples, xsamples, σϕ², intercept);
	print()
end

# ╔═╡ 6fad90c9-e550-4b2e-bfe9-16ae2a576476
Φ

# ╔═╡ e30c2269-d12e-4f37-a3b4-426271bd09e1
md"""

**2) Maximum Likelihood Estimation**

"""

# ╔═╡ 56a6e5de-94de-490b-bce3-a68b02466a1c
begin
	# ML estimation by normal equation
	# wₘₗ = (Φ'*Φ)^(-1)*Φ'*ts
	mle(Φ, t) = (Φ'*Φ)^(-1)*Φ'*t
end

# ╔═╡ 8287a01a-1797-4871-8c96-739f5cad0573
mle(Φ, ts)

# ╔═╡ f08f84e6-ac8f-4ba5-9571-ce7d72de87b9
begin
	plotly()
	# Φₜₑₛₜ is the RBF expanded xs values, from xsamples (1x50) as centers
	Φₜₑₛₜ = basis_expansion(xs, xsamples, σϕ², intercept)
	# tₜₑₛₜ is the true signal
	tₜₑₛₜ = signal_.(xs)
	# θₘₗ is the MLE. It is calculated from Φ which is the xsamples RBF from xsamples. ts is the y values, which contain gaussian noise from the original line
	θₘₗ = mle(Φ, ts)
	plot(xs, tₜₑₛₜ, ylim= [-0.8, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=1.0, markercolor=:gray, markeralpha=1.5, label="Obvs")
	plot!(xs, Φₜₑₛₜ*θₘₗ, linecolor=:green, linestyle=:solid, lw=1, label="ML")
end

# ╔═╡ e7dadcf1-8fe8-45d3-86d3-83b1e4ac081a
size(xs)

# ╔═╡ 3346538f-1f9d-4cd9-bce0-ca496a48637b
md"""
**3) MAP and Ridge regression**

"""

# ╔═╡ 3bbc3aa4-85f1-43de-9940-30515be8d522
md"Poor man's Bayesian approximation"

# ╔═╡ f59fa827-1622-4811-b9aa-3c3d7245d9f5
md"""

The loss is penalised with a $L_2$ norm. So large $\theta$ is discouraged. 

$$L(\theta) = \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \theta^\top \phi_i)^2 + \frac{1}{2\lambda^2} \|\theta\|_2^2 =\frac{1}{2\sigma^2}  (\mathbf y -\Phi\theta)^\top (\mathbf y -\Phi\theta) +  \frac{1}{2\lambda^2} \theta^\top \theta$$

This is the same as applying the likelihood with a zero mean isotropic Gaussian prior, i.e. 

$$p(\theta) = N(0, \lambda^2 I) = \frac{1}{\sqrt{(2\pi)^d |\lambda^2I|}} \text{exp}\left (-\frac{1}{2\lambda^2}\theta^\top \theta \right ).$$ 

The posterior is

$$p(\theta|\mathbf y, \Phi)= \frac{p(\theta)p(\mathbf y|\theta, \Phi)}{p(\mathbf y|\Phi)} \propto p(\theta)p(\mathbf y|\theta, \Phi);$$

Take log and sub-in the definitions of the models and prior:

$$\begin{align}
\ln p(\theta|\mathbf y, \Phi) &= \ln p(\theta) +\ln p(\mathbf y|\theta, \Phi) + C\\
&= -\frac{1}{2\lambda^2}\theta^\top \theta - \frac{1}{2\sigma^2}(\mathbf y-\Phi \theta)^\top (\mathbf y-\Phi \theta)
\end{align}$$


"""

# ╔═╡ 900302fa-5a9c-4e48-af81-fe9589b526a7
md"""

Step by step getting gradient of model + prior:

$\ln p(θ|y,Φ) = - \frac{1}{2σ²}(y-Φθ)^⊤(y-Φθ) - \frac{1}{2λ^2}θ^⊤θ$

Derivative of $(y-Φθ)^⊤(y-Φθ)$ is $2(y-Φθ) (- Φ)$ due to quadratic form, and chain rule.

$= -\frac{1}{\cancel{2}σ^2}\cancel{2}(-Φ^⊤)(y-Φθ) - \frac{1}{\cancel{2}λ^2}\cancel{2}θ$

Cancel 2s:

$= -\frac{1}{σ^2}(-Φ^⊤)(y-Φθ) - \frac{1}{λ^2}θ$

Multiply by $-1$:

$= -\frac{1}{σ^2}Φ^⊤(y-Φθ) + \frac{1}{λ^2}θ$

Then carry on set to 0 and solve for θ.

"""

# ╔═╡ 17fd1c40-c206-4024-8b70-9d5184f23ab4
md"""

To minimise the loss (or maximise the log posterior), take derivative and set to zero we have:

$$\frac{\partial L}{\partial \theta} = - \frac{1}{\sigma^2} \Phi^\top(\mathbf y-\Phi\theta) + \frac{1}{\lambda^2}\theta =0$$

$$\begin{align}&\Rightarrow - \frac{1}{\sigma^2} \Phi^\top \mathbf y+\frac{1}{\sigma^2}  \Phi^\top \Phi\theta + \frac{1}{\lambda^2}\theta =0\\
&\Rightarrow \left (\frac{1}{\sigma^2}\Phi^\top \Phi + \frac{1}{\lambda^2}I\right)\theta = \frac{1}{\sigma^2} \Phi^\top \mathbf y \\
&\Rightarrow \theta_{MAP} = \left (\frac{1}{\sigma^2}\Phi^\top \Phi + \frac{1}{\lambda^2}I\right)^{-1}\frac{1}{\sigma^2} \Phi^\top \mathbf y
\end{align}$$

If we set the penalty coefficient $\frac{1}{\lambda^2}=0$, we recover the MLE. 

Exercise: what is the Hessian ? What if we apply Newton's method?
"""

# ╔═╡ 4b8a60ba-43ca-4121-998f-3f3720ff1bc5
begin
	function ridge_loss(θ, σ², λ², Φ, ts)
		1/(2*σ²) * sum((Φ * θ - ts).^2) + 0.5* (1/λ²) * sum(θ .* θ)
	end

	# The first log pdf is the zero mean gaussian prior
	# The second log pdf is the gaussian model of linear regression
	function logPosterior(θ, σ², λ², Φ, ts)
		logpdf(MvNormal(zeros(size(θ)), sqrt(λ²)), θ) + logpdf(MvNormal(ts, sqrt(σ²)), Φ*θ)
	end
end

# ╔═╡ cb006cf5-d704-4b63-a1b2-0f593541ad16
begin
	λ²_ = 1.0
	σ²_ = 1.0
	l2loss(θ) = ridge_loss(θ, σ²_, λ²_, Φ, ts)
	ForwardDiff.gradient(l2loss, zeros(51))
end

# ╔═╡ ec3f7b50-c33f-4332-830f-88212a453674
begin
	logPost(θ) = logPosterior(θ, σ²_, λ²_, Φ, ts)
	ForwardDiff.gradient(logPost, zeros(51))
end

# ╔═╡ 4565fe08-d0a7-4df5-8b95-f932180dde70
begin
	θ₀ = zeros(size(Φ)[2])
	logPosts = zeros(1500)
	for i in 1:1500
		θ₀ += 0.01 * ForwardDiff.gradient(logPost, θ₀)
		logPosts[i] = logPost(θ₀)
	end
end

# ╔═╡ b0963f2a-cfe7-4443-946b-b55dec626623
θ₀

# ╔═╡ df56b584-8637-49d6-947d-aa5a8318fb63
plot(-logPosts, label="Neg log post")

# ╔═╡ 42043fb8-b0d0-4cc0-983b-d821955023d0
θₘₐₚ(Φ, t, σ², λ²) = inv( (1/σ²) * Φ'*Φ + (1/λ²) * I) * (1/σ²)*Φ'*t

# ╔═╡ 382ac7f8-8814-4e12-a617-8f31704b8a3c
@bind λ² Slider([(0.001:1.0:500)..., Inf])

# ╔═╡ 2bdd44ed-004f-4444-a552-181711176ac8
θ_ridge =θₘₐₚ(Φ, ts, σ²_, λ²)

# ╔═╡ 01eb3620-1451-4c70-a321-4e80aac93dad
begin
	plotly()
	plt = plot(xs, tₜₑₛₜ, ylim= [-0.8, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=1., markercolor=:gray, markeralpha=0.9, label="Obvs")
	plot!(xs, Φₜₑₛₜ*θₘₗ, linestyle=:solid, lw=2, label="ML")
	# λ²s = [0.001, 1.0, 5.0, Inf]
	# for λ²₀ in λ²s
	# 	θ_ridge₀ = θₘₐₚ(Φ, ts, σ²_, λ²₀)
	# 	plot!(xs, Φₜₑₛₜ*θ_ridge₀, linestyle=:solid, lw=1.5, label="MAP λ²="*string(λ²₀))
	# end
	plot!(xs, Φₜₑₛₜ*θ_ridge, linestyle=:solid, lw=1.5, label="MAP λ²="*string(λ²))
	plt
end

# ╔═╡ c0558eae-aaac-457d-aa26-ab758f0f5eec
md"""

**4) Bayesian inference**

Bayesian inference assumes 

* the parameter $\theta$ is random variable rather than the data $\{\mathbf y, \Phi\}$ (we assume $\sigma^2, \lambda^2$ known here for simplicity); 
* and aims at finding the posterior $$p(\theta|\mathbf y,\Phi)$$ as an answer

If we assume the prior $p(\theta)= N(m_0, C_0)$ is Gaussian, we can show that the posterior is also Gaussian; this is called conjugate prior. 

$$\begin{align}
p(\theta|\mathbf y, \Phi) &\propto p(\theta)p(\mathbf y|\theta, \Phi)\\
&= \frac{1}{\sqrt{(2\pi)^d |C_0|}} \text{exp}\left \{-\frac{1}{2} (\theta-m_0)^\top C_0^{-1}(\theta -m_0)\right \}\cdot\\
&\;\;\;\;\;\frac{1}{\sqrt{(2\pi)^N |\Sigma|}} \text{exp}\left \{-\frac{1}{2} (\mathbf y-\Phi\theta)^\top \Sigma^{-1}(\mathbf y -\Phi\theta)\right \};
\end{align}$$
where $\Sigma=\sigma^2 I_N$.

As a function of $\theta$, we can group some terms as constant

$$\begin{align}
p(\theta|\mathbf y, \Phi) &\propto  \text{exp}\left \{-\frac{1}{2} (\theta-m_0)^\top C_0^{-1}(\theta -m_0)\right \}\cdot\\
&\;\;\;\;\; \text{exp}\left \{-\frac{1}{2} (\mathbf y-\Phi\theta)^\top \Sigma^{-1}(\mathbf y  -\Phi\theta)\right \} \\
&= \text{exp}\left \{-\frac{1}{2} (\theta-m_0)^\top C_0^{-1}(\theta -m_0)-\frac{1}{2} (\mathbf y-\Phi\theta)^\top \Sigma^{-1}(\mathbf y  -\Phi\theta)\right \} 
\end{align}$$

**Complete squares** 
Similar to $(ax-b)^2 = a^2 x^2 -2ab x +b^2$, we can expand or complete squares with multivariate quadratic form. E.g. once we identify the quadratic term is e.g. $a^2$ and the linear coefficient is $-2ab$, we know it can form a completed squared $(ax-b)(ax-b)$.

Let's expand a Gaussian's quadratic form (ignore $-1/2$ for the moment):

$$\begin{align}
(\theta-\mu)^\top \Sigma^{-1}(\theta -\mu) &=  \theta^\top \Sigma^{-1}(\theta -\mu) -  \mu^
\top \Sigma^{-1}(\theta -\mu)  \\
&=  \theta^\top \Sigma^{-1}\theta -\theta^\top \Sigma^{-1}\mu -  \mu^
\top \Sigma^{-1}\theta + \mu^
\top \Sigma^{-1}\mu \\
&=\theta^\top \Sigma^{-1}\theta - 2\mu^\top \Sigma^{-1}\theta + C;
\end{align}$$
which implies whenever we identify the following expanded quadratic pattern

$$\theta^\top A\theta + b^\top \theta + c,$$ we can complete the square (which implies the density is Gaussian (check Gaussian's pdf)), and identify the mean and variance as:

$$\begin{cases}\Sigma^{-1} = A \\
b^\top = -2\mu^\top \Sigma^{-1}
\end{cases} \Rightarrow \begin{cases} \mu = -\frac{1}{2} A^{-1} b\\ \Sigma = A^{-1}\end{cases}$$

"""

# ╔═╡ 8cd02480-58c4-4982-b28e-e08750433399
md"""

Now complete the square of the posterior you should find the following expansion:

$$\begin{align}(\theta-m_0)^\top & C_0^{-1}(\theta -m_0)+ (\mathbf y-\Phi\theta)^\top \Sigma^{-1}(\mathbf y  -\Phi\theta) \\
&= \theta^\top C_0^{-1} \theta - 2m_0^\top C_0^{-1} \theta + \theta^\top \Phi^\top \Sigma^{-1} \Phi\theta -2 \mathbf y^\top\Sigma^{-1} \Phi \theta  + c\\
&= \theta^\top (C_0^{-1} +\Phi^\top \Sigma^{-1} \Phi) \theta -2 (m_0^\top C_0^{-1} + \mathbf y^\top\Sigma^{-1} \Phi ) \theta + c
\end{align}$$

Therefore, the posterior should be of a Gaussian form, 

$$p(\theta|\Phi, \mathbf y) = N(m_N, C_N)$$ and 
and the matching mean and variance are

$$\begin{cases}
m_N = -\frac{1}{2} C_N (-2 (m_0^\top C_0^{-1} + y\Sigma^{-1} \Phi )^\top)= C_N( C_0^{-1}m_0 +  \Phi^\top\Sigma^{-1}\mathbf y)\\
C_N = (C_0^{-1} +\Phi^\top \Sigma^{-1} \Phi)^{-1}
\end{cases}$$


The updated posterior mean 

$(C_0^{-1} +\Phi^\top \Sigma^{-1} \Phi)^{-1}( C_0^{-1}m_0 +  \Phi^\top\Sigma^{-1}\mathbf y)$ 
can be viewed as a ''weighted average'' between the prior guess $m_0$ and the MLE estimator $(\Phi\Sigma^{-1}\Phi)^{-1} \Phi^\top \Sigma^{-1}\mathbf y.$ 
"""

# ╔═╡ 290466db-1a98-4062-8aaf-70ed5560d7c7
md"""

Note that the posterior mean ``m_N`` is the same as the MAP estimator once you sub-in 

$$C_0 = \lambda^2 I_d; \Sigma=\sigma^2 I_N;$$

but we have an extra uncertainty term $C_N$ here.

"""

# ╔═╡ 507dac31-2dbe-4264-8efb-43432c5a4fc2
function bayesian_conjugate_lr(Φ, y, Σ, m₀, C₀)
	Cₙ = inv(inv(C₀) + Φ' * inv(Σ) * Φ)
	mₙ = Cₙ * (inv(C₀) * m₀ + Φ'*inv(Σ) * y)
	return MvNormal(mₙ, Symmetric(Cₙ))
end

# ╔═╡ 58b8cf81-194a-4bcd-a0a4-1973b76b54f5
bayesian_conjugate_lr(Φ, ts, σ²_ * I, zeros(size(Φ)[2]), λ²_ *I)

# ╔═╡ b7d4dc52-2c8e-48b1-98f6-4495ce3c4104
zeros(size(Φ)[2])

# ╔═╡ 4e552e4a-3aa7-4502-b087-c96d6711f6a3
md"""

**Bayesian prediction**

As a result, when it comes prediction, given a new test data $\phi_0$, Bayesian predicts it by integration rather than plug-in estimation:

$$\begin{align}p(y_0|\Phi, \mathbf y, \phi_0) &= \int p(y_0, \theta|\Phi,\mathbf y, \phi_0) d{\theta} =\int p(y_0|\theta, \cancel{\Phi,\mathbf y}, \phi_0) p(\theta|\Phi,\mathbf y, \cancel{\phi_0})d\theta\\ 
&= \int p(y_0|\theta, \phi_0) p(\theta|\Phi,\mathbf y)d\theta \\
&= \int N(y_0; \theta^\top \phi_0, \sigma^2) N(\theta; m_N, C_N) d\theta \\
&= N(y_0; m_N^\top \phi_0, \sigma^2_N(\phi_0)),
\end{align}$$ 
where 
$\sigma^2_N(\phi_0) = \sigma^2 + \phi_0^\top C_N \phi_0.$

The integration has used marginal Gaussian's property. Check 2.3.3. of Bishop. But for now you can assume it is correct. This is rare case that we can integrate to a closed form solution. For most general case, we use Monte Carlo method:

$$p(y_0|\Phi, \mathbf y, \phi_0) \approx \frac{1}{M} \sum_{m=1}^M p(y_0|\theta^{(m)}, \sigma^2)= \frac{1}{M} \sum_{m=1}^M N(y_0; \phi_0^\top \theta^{(m)}, \sigma^2);$$ where $\theta^{(m)} \sim N(m_N, C_N).$

it is an average of $M$ Gaussian predictions, which states the essential properties of a Bayesian approach:
* enssemble method: in the sense multiple models all make contribution to the final predictions (each $\theta^{(m)}$ is a model of the ensemble)
* democratic: no unique model dominates the final say (unlike MLE and MAP plug in)


I also want to highlight the difference between prediction of the **mean** signal 

$\phi_0^\top \theta$ at input $\phi_0 = \phi(x_0)$ and prediction of the target or observation itself: $y_0$. The former has the following posterior distribution 

$$\phi_0^\top \theta|y,\Phi \sim N(\phi_0^\top m_N, \phi_0^\top C_N \phi_0).$$ We have used the following property every linear combination of a Gaussian is still a Gaussian and its mean $E[\phi_0^\top \theta | \mathbf y, \Phi] = \phi_0^\top E[\theta|\mathbf y, \Phi] = \phi_0^\top m_N$ and variance $\text{Var}[\phi_0^\top \theta|\mathbf y, \Phi] = \phi_0^\top \text{Var}[\theta|\mathbf y, \Phi] \phi_0$.

The predictive distribution of the target itself $y_0 = \phi_0^\top \theta + \sigma^2$ has another layer of uncertanty, the observation noise $\sigma^2$. Naturally, the predictive distribution should have wider prediction intervals or equivalently larger uncertainty:

$$y_0|\mathbf y,\Phi \sim N(\phi_0^\top m_N, \phi_0^\top C_N \phi_0 +\sigma^2)$$
"""

# ╔═╡ be9aaf3a-8104-442a-b77b-c1c1a25a4a8d
begin
	plotly()
	plt2 = plot(xs, tₜₑₛₜ, ylim= [-1.5, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.0, markerstrokewidth=1.0, markercolor=:gray, markeralpha=0.9, label="Obvs")
	plot!(xs, Φₜₑₛₜ*θₘₗ, linestyle=:solid, lw=2, label="ML")
	mc = 100
	postModel=bayesian_conjugate_lr(Φ, ts, σ²_ * I, zeros(size(Φ)[2]), λ²_ *I)
	Θ = rand(postModel, mc)
	for m in 1:mc
		plot!(xs, Φₜₑₛₜ*Θ[:, m], alpha=0.2 ,color=:red, lw=0.1, label="")
	end
	preds = Φₜₑₛₜ*Θ
	mc_ts = mean(preds, dims=2)
	plot!(xs, mc_ts, alpha=1 ,color=:red, lw=2, label="Bayesian MC mean")
	plot!(xs, Φₜₑₛₜ * postModel.μ, alpha=1, color=:blue, lw=2, label="Bayesian theory mean")
	vs = ([Φₜₑₛₜ[i, :]' *postModel.Σ * Φₜₑₛₜ[i, :] for i in 1:size(Φₜₑₛₜ)[1]])
	plot!(xs, (mc_ts + 1.645 * sqrt.(vs)) , alpha=1, color=:blue, lw=1, label="[5%, 95%] CI mean")
	plot!(xs, (mc_ts - 1.645 * sqrt.(vs)) , alpha=1, color=:blue, lw=1, label="")
	plot!(xs, (mc_ts + 1.645 * sqrt.(vs .+σ²_)) , alpha=1, color=:green, lw=1, label="[5%, 95%] CI y₀")
	plot!(xs, (mc_ts - 1.645 * sqrt.(vs .+ σ²_)) , alpha=1, color=:green, lw=1, label="")
	plt2
end

# ╔═╡ bd00bba5-a936-479f-bf0b-8b3a1fa920ef
vs

# ╔═╡ 06d29e89-07bc-48a1-ae4d-a66e573ed270
md"""

**5) Evidence procedure**

"""

# ╔═╡ e1891b72-4cc8-4209-bc88-ac528d38bca8
md"""

We have used $\sigma^2=\lambda^2=1.0$ for the previous section. They are just some random guess which clearly does not fit the data well. The evidence procedure can optimise the hyperparameter in a principled way rather than resorting to cross validation.

The idea is to treat $\sigma^2, \lambda^2$ as two parameters to optimise.  Here we are basically combining a Bayesian and frequentist approach together:
* infer $\theta$ in a Bayesian way; i.e. find its posterior conditional on $\sigma^2, \lambda^2$ 
* but treat $\sigma^2, \lambda^2$ as hyperparameters and to optimise them.

A more Bayesian approach however is to treat them as two additional random variables and infer them in the same way as $\theta$.

Bishop Chapter 3.5.1 and 3.5.2 has used a direct optimisation approach to optimise $\lambda^2$ and $\sigma^2$; the estimation formula are

$$\hat{\sigma^2} = \frac{1}{N-\gamma} \|\mathbf y - \Phi m_N\|_2^2,\; \hat{\lambda^2} = \frac{m_N^\top m_N}{\gamma},$$ where 

$\gamma = \sum_{i} \frac{\nu_i}{\frac{1}{\lambda^2} + \nu_i},$ where $\nu_i$ is the eigen value of $\frac{1}{\sigma^2} \Phi^\top \Phi$ (with eigen vector $\boldsymbol v_i$). 

$$\left (\frac{1}{\sigma^2} \Phi^\top \Phi\right ) \boldsymbol v_i= \nu_i \boldsymbol v_i$$


Check the book for details of the derivation. But the idea is to find the model evidence first:

$$p(\mathbf y| \lambda^2, \sigma^2, \Phi)$$

which can be viewed as the likelihood for parameters $\lambda^2, \sigma^2$. In essense it becomes a MLE for $\lambda^2, \sigma^2$. The model evidence is calculated via


$$\begin{align} p(\mathbf y| \lambda^2, \sigma^2, \Phi) &= \int p(\mathbf y, \theta| \lambda^2, \sigma^2, \Phi) d\theta \\
&= \int p(\theta| \lambda^2, \cancel{\sigma^2, \Phi}) p(\mathbf y|\theta, \cancel{\lambda^2}, \sigma^2, \Phi)d\theta \\
&= \int p(\theta| \lambda^2) p(\mathbf y|\theta,  \sigma^2, \Phi)d\theta 
\end{align}$$ where the integration has an analytical form. 


The above integration is very similar to what we did in the complete square section: however we cannot throw away those constant terms (we are investigating the evidence as a function of $\sigma^2, \lambda^2$, which were viewed as constants when we derive the posterior of $\theta$).
"""

# ╔═╡ 46571173-b0b3-4b17-92b6-8e83012c6e64
md"""

Note that 

$\nu_i = \frac{1}{\sigma^2} \nu_i^0,$ where $\nu_i^0$ is the eigen value of $\Phi^\top \Phi$:

$$\Phi^\top \Phi \boldsymbol v_i =  \nu_i^0 \boldsymbol v_i,$$ therefore

$$\frac{1}{\sigma^2}\Phi^\top \Phi \boldsymbol v_i = \frac{1}{\sigma^2}\nu_i^0 \boldsymbol v_i;$$

We can eigen-decompose $\Phi^\top \Phi$ once and rescale its eigen values. 
"""

# ╔═╡ 95544acb-1cd3-41a6-aafb-43954a957ac6
begin
	function evidence_procedure(Φ, ts; tol= 1e-4)
		ν₀, vvs=eigen(Φ'* Φ)
		N = length(ts)
		σ²₀, λ²₀=  1.0, 1.0
		iters = 1
		for i in 1:1000
			post = bayesian_conjugate_lr(Φ, ts, σ²₀ * I, zeros(size(Φ)[2]), λ²₀ *I)
			mₙ, Cₙ = post.μ, post.Σ
			sse₁ = sum((ts - Φ*mₙ).^2)
			sse₂ = sum(mₙ .* mₙ)
			ν = (1/σ²₀) * ν₀
			γ = sum(ν ./ (1/λ²₀ .+ ν))
			σ²₀_ = sse₁/(N -γ) 
			λ²₀_ = sse₂/γ
			if (abs(σ²₀_ - σ²₀) + abs(λ²₀_ - λ²₀)) < tol
				σ²₀, λ²₀ = σ²₀_, λ²₀_
				break
			end
			σ²₀, λ²₀ = σ²₀_, λ²₀_
			iters += 1
		end
		post = bayesian_conjugate_lr(Φ, ts, σ²₀ * I, zeros(size(Φ)[2]), λ²₀ *I)
		return σ²₀, λ²₀, post, iters
	end
end

# ╔═╡ 5b844308-b341-49e5-9473-b65b56e26880
σ²₀, λ²₀, post_ep, iters =evidence_procedure(Φ, ts)

# ╔═╡ 8dcdbe4a-8159-49c4-b658-2fef6a4b4ff9
begin
	plotly()
	plt3 = plot(xs, tₜₑₛₜ, ylim= [-1.5, 2.5], linecolor=:black, linestyle=:dash, label="Original")
	scatter!(xsamples, ts, markershape=:xcross, markersize=2.5, markerstrokewidth=1.0, markercolor=:gray, markeralpha=0.9, label="Obvs")
	plot!(xs, Φₜₑₛₜ * θₘₗ, linestyle=:solid, lw=2, label="ML")
	plot!(xs, Φₜₑₛₜ * post_ep.μ, alpha=1, color=:blue, lw=2, label="Evidence")
end

# ╔═╡ ba0ec305-a63b-4686-9365-ebbc7a04e899
md"""

Another approach to optimise $\lambda^2, \sigma^2$ is to use **EM** algorithm. We treat $\theta$ as the unobserved hidden random variable of the EM procedure. In the E step, we find the expected complete log data likelihood (CDLL)

$$\begin{align}\mathcal L(\lambda^2, \sigma^2) =E[\ln p(\theta, \mathbf y |\Phi, \sigma^2, \lambda^2)] &= E[\ln p(\theta|\lambda^2) +\ln p(\mathbf y|\Phi, \sigma^2, \theta)] ,
\end{align}$$
where the expectation is taken with the posterior $p(\theta|\Phi, \mathbf y) = N(m_N, C_N).$

In the M step, we optimise the expected CDLL w.r.t to $\lambda^2, \sigma^2$. It is left as an exercise to derive and implement the algorithm. You may need the following identity to derive the EM algorithm: 

$$E_{p(\theta|\Phi, \mathbf y)}[\theta^\top \theta] =m_N^\top m_N + C_N;$$ which can be proved as 

$$\text{Var}_{p(\theta|\Phi, \mathbf y)}[\theta] = E[\theta^\top\theta] - E[\theta]^\top E[\theta] \Rightarrow  C_N = E[\theta^\top\theta] - m_N^\top m_N\Rightarrow E[\theta^\top\theta] = m_N^\top m_N + C_N$$
"""

# ╔═╡ becbf88b-47cd-4d06-a4dd-56cbddc96dd6
md"""
## Part 2: Regularisation

Frequentist ML deals with overfitting by adding a penalty term to the loss. It actually corresponds to optimising a posterior distribution, which has its root in Bayesian inference:

$$p(\beta|X, y) \propto p(\beta)p(y|\beta, X).$$

If we assume the prior $p(\beta)$ is zero mean Gaussian, then maximising $p(\beta|X,y)$ recovers ridge regression:

$$\beta_{\text{ridge}} = \beta_{\text{MAP}} =\arg\max_\beta p(\beta|X,y)$$

Given following dataset, a binary classification problem
1. use Newton's method to find the MLE of the logistic regression.
    * for linearly seperable note that the larger the norm $\|\beta_t\|$ is, the larger the likelihood is; what does this imply?
2. use Newton's method to find the MAP, *i.e.* Ridge logistic regression.
3. plot the prediction as a surface or contour plot; $$p(y|\boldsymbol x,\hat \beta)$$ as a function of $\boldsymbol x\in R^2$; are they fundamentally different?
"""

# ╔═╡ 41a5e342-eb5a-4ff7-847d-72c0e0ee96f4
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	fig = plot(D1[:,1], D1[:,2], xlabel=L"$x_1$", ylabel=L"x_2", label="class 1", seriestype=:scatter, ratio=1.0, title="Linearly Separable Data for Logistic Regression")
	plot!(D2[:,1], D2[:,2], xlabel=L"$x_1$", ylabel=L"x_2", label="class 2", seriestype=:scatter, ratio=1.0)
	# scatter!(D2[:,1], D2[:,2], label="class 0")
	savefig(fig, "logistic_regression.png")
end

# ╔═╡ 08e51e0e-d4c5-4464-82cb-105e81b527d9
md"""


4. Use Laplace method to approximate the posterior; 
   * plot the prediction again using Monte carlo method: note you need to sample from the Laplace approximated posterior;

"""

# ╔═╡ 7cdbd3cd-2fd4-4962-9c66-b3aff7559fc7
md"""
**Solution**

**1) MLE by Newton's method** We have derived the gradient and Hessian for logistic regression last time. I will just copy them over here.



$$g(\boldsymbol \beta) \triangleq\nabla L(\boldsymbol \beta)  =  \mathbf{X}^\top(\mathbf{y}- \boldsymbol{\sigma}),$$ where $\boldsymbol{\sigma} = \begin{bmatrix}\sigma(x_1 ^\top \boldsymbol{\beta})\\ \sigma(x_2 ^\top \boldsymbol{\beta})\\ \vdots\\\sigma(x_n ^\top \boldsymbol{\beta})\end{bmatrix}.$

$$H(\boldsymbol \beta) \triangleq \nabla\nabla L(\boldsymbol \beta)=\mathbf{X}^\top \mathbf{D} \mathbf{X},$$ where 

$\mathbf{D} = \text{diag} (\sigma_i(\sigma_i-1))$ for $i =1,\ldots,N$.
"""

# ╔═╡ d3caa211-322b-4f27-9e4a-b5aa18181b11
function logRegLogLiks(y, X, β)
	σ = logistic.(X*β)
	grad = X' * (y - σ)
	H = X' * (σ .* (σ .-1) .* X)
# 		you may also write it as derived by creating a diagonal matrix D
	# H = X' * diagm(σ .* (σ .-1)) * X)
	sum(logpdf.(Bernoulli.(σ), y)), grad, H
end	

# ╔═╡ cab19a19-2736-4b36-98be-12b0da0976ab
md"""
**2) MAP by Newton's method**

We need to find the gradient and Hessian for the log posterior. 

$$\ln p(\boldsymbol \beta|\mathbf{X, y}) = \ln p(\boldsymbol\beta) + \ln p(\mathbf y|\boldsymbol\beta, \mathbf X)$$


The gradient (the same applies to Hessian) is 

$$\nabla \ln p(\boldsymbol \beta|\mathbf{X, y}) = \nabla \ln p(\boldsymbol\beta) + \nabla \underbrace{p(\mathbf y|\boldsymbol\beta, \mathbf X)}_{L(\boldsymbol\beta)}.$$


We only need to find the gradient and Hessian of the prior and add them to the likelihood's gradient and Hessian.

The prior is a Gaussian with mean and variance $m_0, C_0$, therefore

$$\begin{align}\ln p(\boldsymbol \beta) &= \ln\left \{\frac{1}{\sqrt{(2\pi)^d |C_0|}} \text{exp}\left (-\frac{1}{2}(\boldsymbol \beta^\top -m_0)^\top C_0^{-1}(\boldsymbol \beta -m_0) \right )\right \} \\
&= -\frac{d}{2} \ln 2\pi -\frac{1}{2}\ln |C_0|-\frac{1}{2}(\boldsymbol \beta^\top -m_0)^\top C_0^{-1}(\boldsymbol \beta -m_0) 
\end{align}$$
"""

# ╔═╡ 87afe09c-8426-4588-b319-132c5adb01a8
md"""
Drew Note
What's happening on the LHS above is:

$\ln{\left( \frac{1}{\sqrt{(2π)ᵈ|C₀|}} \right)}$

Split into two:

$\ln{\left( \frac{1}{\sqrt{(2π)ᵈ}} \frac{1}{\sqrt{|C₀|}}\right)}$

Convert fraction to negative exponents:

$\ln{\left(2π^{-\frac{d}{2}} * |C_0|^{-\frac{1}{2}}\right)}$

Push logs in:

$\ln(2π^{-\frac{d}{2}}) + \ln(|C_0|^{-\frac{1}{2}})$

Move powers down:

$-\frac{d}{2} \ln(2π) + -\frac{1}{2} \ln(|C_0|)$

And the rest is clear.

$= -\frac{d}{2} \ln 2\pi -\frac{1}{2}\ln |C_0|-\frac{1}{2}(\boldsymbol \beta^\top -m_0)^\top C_0^{-1}(\boldsymbol \beta -m_0)$
"""

# ╔═╡ c3f85d58-bfee-4115-9983-9831596e2a74
md"""
Therefore,

$$\nabla \ln p(\boldsymbol\beta) = -(C_0^{-1})^\top (\boldsymbol \beta -m_0)= -C_0^{-1} (\boldsymbol \beta - m_0)$$

The Hessian is 

$$\nabla\nabla \ln p(\boldsymbol\beta) = - C_0^{-1} I = - C_0^{-1}$$

The gradient for the log posterior therefore is 

$$g(\boldsymbol \beta) = -C_0^{-1} (\boldsymbol \beta - m_0) + \mathbf{X}^\top(\mathbf{y}- \boldsymbol{\sigma}),$$ the same applies to Hessian. 
"""

# ╔═╡ df407e76-93c5-40e9-b633-5c3922a96ab3
begin
	function logPosteriorLogisticR(w, m0, C0, X, y)
		σ = logistic.(X * w)
		Λ0 = inv(C0)
		grad = - Λ0  * (w-m0) + X' * (y - σ)
		d = σ .* (σ .- 1)
		H = (X .* d)' * X - Λ0
		# return logpdf(MvNormal(m0, V0), w) + sum(logpdf.(Bernoulli.(σ), y)), grad, H
		return -0.5* (w-m0)' * Λ0 * (w-m0) + sum(logpdf.(Bernoulli.(σ), y)), grad, H
	end
	
end

# ╔═╡ cfe2b3e6-957a-4807-b4f9-80f9d465ba44
begin
	dim = 2+1
	x00 = zeros(dim)
	m0 = zeros(dim)
	C0 = 1000. .* Matrix(I, dim, dim)
	logPosteriorLogisticR(x00, m0, C0, D, targets)

end

# ╔═╡ 3ca475e5-a56d-49f1-94db-b51faf923e0f
ForwardDiff.hessian((x) -> logpdf(MvNormal(m0,C0), x), zeros(3)) 

# ╔═╡ f0c17f6a-2856-4de9-9981-d6d25a544e4f
md"""

Gradient check with finite difference
"""

# ╔═╡ ea3775a0-9929-4b5a-ae06-2a343fe132f7
rstfd = FiniteDifferences.grad(central_fdm(5,1), (x) -> logPosteriorLogisticR(x, m0, C0, D, targets)[1], x00)[1]

# ╔═╡ 82887c60-5d63-4d3f-bf4c-a89ee94bb1ad
md"You can also use auto-diff:"

# ╔═╡ 5b9514f6-acd8-4be7-9741-18e58c9377d7
begin
	g_(x) = Zygote.gradient((x_) -> logPosteriorLogisticR(x_,  m0, C0, D, targets)[1], x)[1] 
	g_(x00)
end

# ╔═╡ 9fcaec3f-e442-4150-a759-2b78376b6f98
# Zygote doesn't work for this due to a bug; have to write logpdf of Gaussian manually to make it work
# ForwardDiff.hessian((x) -> logPosteriorLogisticR(x, m0,  V0, D, targets)[1], x00) 
Zygote.hessian((x) -> logPosteriorLogisticR(x, m0,  C0, D, targets)[1], x00) 

# ╔═╡ fff9c3a4-673d-41b9-828e-1c46e527c51b
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

# ╔═╡ e66e149b-4c0a-4ecb-b793-d270777ab1a2
begin
	ww = randn(dim) * 0.25
	η = 0.02
	iters_ = 20000
	opt = Descent(η)
	# logLikGrad(x) = Zygote.gradient((w) -> logRegLogLiks(targets, D, w)[1], x)
	logLiks_MLE = zeros(iters_)
	wws = Matrix(undef, 3, iters_)
	for i in 1:iters_
		logLiks_MLE[i], gt, _ = logRegLogLiks(targets, D, ww)
		Flux.Optimise.update!(opt, ww, -gt)
		wws[:, i] = ww 
	end
end

# ╔═╡ ce736d70-6e43-4595-bf90-bea0497e0655
begin
	plotly()
	plot(logLiks_MLE, label="Log lik")
end

# ╔═╡ ec23667b-13c5-4ced-b218-215b846825fc
begin
	plotly()
	plot(wws[1,1:5:end], label="w0")
	plot!(wws[2,1:5:end], label="w1")
	plot!(wws[3,1:5:end], label ="w2")
end

# ╔═╡ e6e48303-ef1d-4e8d-84c9-077b693fd39b
begin
	plotly()
	logReg(x, w, w0) = logistic(x' * w + w0)	
	x1 = range(1, stop=10, length=100)
	x2 = range(1, stop=10, length=100)
	plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], ww[2:3], ww[1]), st=:contour, legend = false, label="contour of the prediction")
end

# ╔═╡ b565f3dd-6f3d-4a9f-9e25-7038ced1b378
begin
	wₘₗ, mle_history = mleNewton((x) -> logRegLogLiks(targets, D, x), zeros(dim); tol = 1e-5, maxIters=1000)
	wₗ₂, l2_history = mleNewton((x) -> logPosteriorLogisticR(x, m0, C0, D, targets), zeros(dim); tol = 1e-5, maxIters=1000)
end

# ╔═╡ 0c57e155-40e1-4e5c-9960-66c2fe1d7e86
begin
	#plotly()
	gr()
	fig2 = plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], wₘₗ[2:3], wₘₗ[1]), st=:surface, legend = false, label="MLE", title="Logistic Regression with MLE")
	savefig(fig2, "logistic_regression_mle.png")
end

# ╔═╡ 8a1b9517-5645-42e0-8e39-17f0d66ec7b1
begin
	fig3 = plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], wₗ₂[2:3], wₗ₂[1]), st=:surface, legend = false, label="MAP", title="Logistic Regression with MAP")
	savefig(fig3, "logistic_regression_map.png")
end

# ╔═╡ a061819a-391a-405b-b833-a398e3d279d1
begin
	gr()
	wws_ = wws[:, 1:1000:20000]
	anim = @animate for i = 1:size(wws_)[2]
		plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], wws_[2:3, i], wws_[1, i]), st=:contour)
	end
	gif(anim, "ws.gif", fps = 5)
end

# ╔═╡ cfdae888-3226-4230-979a-b02dd5b31ecd
PlutoUI.Resource("https://i.imgur.com/8V5pwwa.gif")

# ╔═╡ 14d4ed18-4fa5-4237-a57e-08014f4c589d
begin
	plot(mle_history, label="MLE")
	plot!(l2_history, label="MAP")
end

# ╔═╡ afb1343d-06c6-4450-be00-fee1e6ca1f44
begin
	wwₗ₂ = zeros(dim) 
	ηₗ₂ = 0.01
	# iters = 20000
	optₗ₂ = Descent(ηₗ₂)
	# logLikGradl2(x) = Zygote.gradient((w) -> logPosteriorLogisticR(w, m0, V0, D, targets)[1], x)
	logLiksₗ₂ = zeros(iters_)
	wwsₗ₂ = Matrix(undef, 3, iters_)
	for i in 1:iters_
		logLiksₗ₂[i], gt, _ = logPosteriorLogisticR(wwₗ₂, m0, C0, D, targets)
		Flux.Optimise.update!(optₗ₂, wwₗ₂, -gt)
		wwsₗ₂[:, i] = wwₗ₂
	end
end

# ╔═╡ d4d02d50-376c-4bec-a72f-35d5d57eda2c
# MLE Newton
wₘₗ

# ╔═╡ a28d1e12-84bd-404f-a502-3d8d6809b032
# MLE Gradient Descent
ww

# ╔═╡ 97c1146f-4803-42f2-9ca2-5258435c28aa
# MAP Gradient Descent
wwₗ₂

# ╔═╡ 615d7bd8-e388-46eb-9860-deff32126b6b
begin
	plotly()
	plot(wws[1,1:5:end], label="w0")
	plot!(wwsₗ₂[1,1:5:end], label="w0ₗ₂")
	plot!(wws[2,1:5:end], label="w1")
	plot!(wwsₗ₂[2,1:5:end], label="w1ₗ₂")
	plot!(wws[3,1:5:end], label ="w2")
	plot!(wwsₗ₂[3,1:5:end], label ="w2ₗ₂")
end

# ╔═╡ d347a153-ee3d-4265-aa24-7b0cda95fcb3
begin
	gr()
	wwsₗ₂_ = wwsₗ₂[:, 1:1000:20000]
	animₗ₂ = @animate for i = 1:size(wwsₗ₂_)[2]
		plot(x1, x2, (xx1,xx2)-> logReg([xx1, xx2], wwsₗ₂_[2:3, i], wwsₗ₂_[1, i]), st=:contour)
	end
	gif(animₗ₂, "wsl2.gif", fps = 5)
end

# ╔═╡ 105a6c92-c85f-4b71-9a5a-8cf5f0fc59f5
md"""

**4) Laplace approximation**

First of all, why we need to use Laplace approximation ? It is because the posterior is no longer in closed analytical form. Unlike linear regression, the posterior $p(\boldsymbol \beta|\mathbf{X,y})$ is still of Gaussian form; the posterior for logistic regression is not longer Gaussian (we cannot apply the complete square technique due to the logistic function in the likelhood term).


**Laplace approximation transforms the Bayesian inference problem to an optimisation problem.** The benefit is that most result we have derived is ready to be reused.

The idea is to apply Taylor's second order expansion on the log-posterior function at the maximum, i.e. MAP estimator, then the quadratic function of the Taylor approximation naturally becomes our Gaussian approximation:

$$\hat p(\boldsymbol \beta|\mathbf{X,y}) = N(\boldsymbol \beta; \boldsymbol \beta_{MAP}, \mathbf H^{-1}),$$ where $\mathbf H$ is evaluated at the MAP estimator.

As a by-product, we also have an approximation of the model evidence (check MLAPP P.258) but it is due to Gaussian's normalising constant.

$p(\mathbf y|\mathbf X) \approx e^{-E(\mathbf \beta_{MAP})} (2\pi)^{d/2} |\mathbf H|^{-\frac{1}{2}}.$

Where 

$E(\boldsymbol \beta) = -\ln p(\boldsymbol \beta) - \ln p(\boldsymbol \beta|\mathbf{X,y})$ the negative of the (unnormalised) log posterior.


By using this result, we can easily rediscover the evidence procedure for linear regression actually. Note that the Laplace approximation for the linear regression model is exact as the log posterior itself is quadratic function (its quadratic approximation is exact). 

More importantly and interestingly, we can apply evidence procedure here to optimise $C_0 = \lambda^2 I$ in a principled way. It is also left as an exercise. 


Laplace approximation method can be easily extended to Bayesian Neural Networks. We apply gradient descent first to find a stationary point, then we need to calculate the Hessian. Then we have the Laplace approximated posterior for the BNN.
"""

# ╔═╡ 9155e950-6608-415a-b932-78ff67e1b4bd
begin
	
	function LaplaceLogisticR(X, y, dim; m0= zeros(dim), V0 = 100 .* Matrix(I, dim, dim), maxIters = 1000, tol= 1e-4)
		wt = zeros(dim)
		fts = zeros(maxIters)
		Ht = zeros(dim, dim)
		postLRFun(x) = logPosteriorLogisticR(x, m0, V0, X, y)
		# use Newton's method to speed up the optimisation
		for t in 1:maxIters
			fts[t], gt, Ht = postLRFun(wt)
			wt = wt - Ht\gt
			if t > 1 && abs(fts[t] - fts[t-1]) < tol
				fts = fts[1:t]
				break
			end
		end
		# inverse of the negative Hessian is used as the approximating Gaussian's variance
		V = Hermitian(inv(-1*Ht))
		return wt, V, fts
	end
end

# ╔═╡ 58aa834c-db17-4274-829c-669d393fbbc6
ME = 1e2 .* Matrix(I, dim, dim)

# ╔═╡ 50e605ea-37b3-49ec-802f-9f7a7bf04a20
mLP, VLP, _ = LaplaceLogisticR(D, targets, dim; m0=m0, V0 = ME);

# ╔═╡ 53047746-8e91-4899-9441-2d2127ffe7f2
VLP

# ╔═╡ 6b077bbb-c8a0-4d06-b8d9-adff66d90fb2
mLP

# ╔═╡ 5b648ec8-d3fd-48cf-9ec4-083eb0363303
md"""
**Bayesian predictive distribution**

$$P(y_{n+1}|x_{n+1}, D) = \int P(y_{n+1} |x_{n+1}, w)P(w|D) dw$$

Its Monte Carlo estimator is 

$$P(y_{n+1}|x_{n+1}, D) \approx \frac{1}{M} \sum_{i=1}^{M} P(y_{n+1}|x_{n+1}, w^{(i)})$$

1. What is $P(y_{n+1}|x_{n+1}, w^{(i)})$ ? it is just Bernoulli. 

2. Comparing with frequentist's plugin point estimator, $P(y_{n+1}|x_{n+1}, w_{\text{MAP}})$, what is the difference ? 

"""

# ╔═╡ 165ccdf9-ae5b-42c9-b372-ef8676a07697
begin
	function predictiveLogLik(ws, X, y)
		mc = size(ws)[1]
		h = logistic.(X * ws')
		logLiks = sum(logpdf.(Bernoulli.(h), y), dims=1)
		log(1/mc) + logsumexp(logLiks)
	end
	
	
	function mcPrediction(ws, X)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.(X * ws), dims=2)
	end
	
	function mcPrediction(ws, x1, x2)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.([1.0 x1 x2] * ws))
	end
end

# ╔═╡ d348d150-be97-43fc-a9ca-dc3e9e6960af
begin
# 	Frequentist MAP estimator prediction
	gr()
	ppfLRPoint(x, y) = mcPrediction(wwₗ₂, x, y)
	contour(x1, x2, ppfLRPoint, xlabel=L"$x_1$", ylabel=L"x_2", fill=true, connectgaps=true, line_smoothing=0.85, title="L2 MAP prediction", c=:roma)
	scatter!(D1[:,1], D1[:,2], label = "class 1")
	scatter!(D2[:,1], D2[:,2], label = "class 2")
end

# ╔═╡ 1cba7665-1745-49e6-8fb6-72d6373b7d3f
begin
	gr()
 	mcLP = rand(MvNormal(mLP, Matrix(VLP)), 10000)
	ppfLP(x, y) = mcPrediction(mcLP, x, y)
	contour(x1, x2, ppfLP, xlabel=L"$x_1$", ylabel=L"x_2", fill=true,   connectgaps=true, line_smoothing=0.85, title="Laplace Approximation Monte Carlo Prediction", c=:roma)
	scatter!(D1[:,1], D1[:,2], label ="class 1")
	scatter!(D2[:,1], D2[:,2], label = "class 2")
end

# ╔═╡ 6048782e-3103-4e72-8a86-476e28b0a623
VLP

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Distributions = "~0.25.62"
DistributionsAD = "~0.6.40"
FiniteDifferences = "~0.12.24"
Flux = "~0.13.3"
ForwardDiff = "~0.10.30"
LaTeXStrings = "~1.3.0"
Plots = "~1.29.1"
PlutoUI = "~0.7.39"
StatsFuns = "~1.0.1"
StatsPlots = "~0.14.34"
Zygote = "~0.6.40"
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

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "8dade591a24870ab163e2dd13900c2085e0f805c"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.16"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "7ec9c9e30db6167ec8a38ca2d1bbd40179a9014f"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.11"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "e4e5ece72fa2f108fb20c3c5538a5fa9ef3d668a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.11.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "34e265b1b0049896430625ce1638b2719c783c6b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.35.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

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
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

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

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

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
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "d32a7e392ca7fb144927ab1a1d59b4bab681266e"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

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

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

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
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

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

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "0ee1275eb003b6fc7325cb14301665d1072abda1"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.24"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "62350a872545e1369b1d8f11358a21681aa73929"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.3"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

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

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "21b5d9da260afa6a8638ba2aaa0edbbb671c37bd"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.0"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

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

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

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

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "Random", "ShowCases", "Statistics", "StatsBase"]
git-tree-sha1 = "c92a10a2492dffac0e152a19d5ffd99a5030349a"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.1"

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

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "a0331452b4cfd5e53ee2325376794aea47364d5a"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.7"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "e161b835c6aa9e2339c1e72c3d4e39891eac7a4f"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.3"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "ec2e30596282d722f018ae784b7f44f3b88065e4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.6"

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
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "013596dcee5e55eb36ff56b8d4df888df01e040d"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.6"

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
git-tree-sha1 = "3e32c8dbbbe1159a5057c80b8a463369a78dd8d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.12"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "9e42de869561d6bdf8602c57ec557d43538a92f0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.1"

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

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

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

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

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
git-tree-sha1 = "69fa1bef454c483646e8a250f384e589fd76562b"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.6"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "5d2c08cef80c7a3a8ba9ca023031a85c263012c5"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2bbd9f2e40afd197a1379aef05e0d85dba649951"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.7"

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
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

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

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "464d64b2510a25e6efe410e7edab14fffdc333df"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.20"

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

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

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
# ╟─f5a442b0-eb0c-11ec-183c-556140c4a65e
# ╟─10cfeab1-5115-4473-98be-a1e678432762
# ╠═1933425b-3d88-44d1-9cda-a600b5863d1e
# ╟─52352f36-9ff3-4f60-afca-880c3a7b1214
# ╠═984f132a-9edc-4a15-a3da-811c73d032b5
# ╠═0a72dd30-b0e8-4750-a6e9-add45e7f8f36
# ╠═330b8041-c4c7-43ca-abde-3d65107c5377
# ╠═0ea6f598-7d8b-4d4a-95a9-46abfdd655ee
# ╠═6fad90c9-e550-4b2e-bfe9-16ae2a576476
# ╟─e30c2269-d12e-4f37-a3b4-426271bd09e1
# ╠═56a6e5de-94de-490b-bce3-a68b02466a1c
# ╠═8287a01a-1797-4871-8c96-739f5cad0573
# ╠═f08f84e6-ac8f-4ba5-9571-ce7d72de87b9
# ╠═e7dadcf1-8fe8-45d3-86d3-83b1e4ac081a
# ╟─3346538f-1f9d-4cd9-bce0-ca496a48637b
# ╟─3bbc3aa4-85f1-43de-9940-30515be8d522
# ╟─f59fa827-1622-4811-b9aa-3c3d7245d9f5
# ╟─900302fa-5a9c-4e48-af81-fe9589b526a7
# ╟─17fd1c40-c206-4024-8b70-9d5184f23ab4
# ╠═4b8a60ba-43ca-4121-998f-3f3720ff1bc5
# ╠═cb006cf5-d704-4b63-a1b2-0f593541ad16
# ╠═ec3f7b50-c33f-4332-830f-88212a453674
# ╠═4565fe08-d0a7-4df5-8b95-f932180dde70
# ╠═b0963f2a-cfe7-4443-946b-b55dec626623
# ╟─df56b584-8637-49d6-947d-aa5a8318fb63
# ╠═42043fb8-b0d0-4cc0-983b-d821955023d0
# ╠═382ac7f8-8814-4e12-a617-8f31704b8a3c
# ╠═2bdd44ed-004f-4444-a552-181711176ac8
# ╟─01eb3620-1451-4c70-a321-4e80aac93dad
# ╟─c0558eae-aaac-457d-aa26-ab758f0f5eec
# ╟─8cd02480-58c4-4982-b28e-e08750433399
# ╟─290466db-1a98-4062-8aaf-70ed5560d7c7
# ╠═507dac31-2dbe-4264-8efb-43432c5a4fc2
# ╠═58b8cf81-194a-4bcd-a0a4-1973b76b54f5
# ╠═b7d4dc52-2c8e-48b1-98f6-4495ce3c4104
# ╟─4e552e4a-3aa7-4502-b087-c96d6711f6a3
# ╠═be9aaf3a-8104-442a-b77b-c1c1a25a4a8d
# ╠═bd00bba5-a936-479f-bf0b-8b3a1fa920ef
# ╟─06d29e89-07bc-48a1-ae4d-a66e573ed270
# ╟─e1891b72-4cc8-4209-bc88-ac528d38bca8
# ╟─46571173-b0b3-4b17-92b6-8e83012c6e64
# ╠═95544acb-1cd3-41a6-aafb-43954a957ac6
# ╠═5b844308-b341-49e5-9473-b65b56e26880
# ╟─8dcdbe4a-8159-49c4-b658-2fef6a4b4ff9
# ╟─ba0ec305-a63b-4686-9365-ebbc7a04e899
# ╟─becbf88b-47cd-4d06-a4dd-56cbddc96dd6
# ╠═41a5e342-eb5a-4ff7-847d-72c0e0ee96f4
# ╟─08e51e0e-d4c5-4464-82cb-105e81b527d9
# ╟─7cdbd3cd-2fd4-4962-9c66-b3aff7559fc7
# ╠═d3caa211-322b-4f27-9e4a-b5aa18181b11
# ╟─cab19a19-2736-4b36-98be-12b0da0976ab
# ╟─87afe09c-8426-4588-b319-132c5adb01a8
# ╟─c3f85d58-bfee-4115-9983-9831596e2a74
# ╠═df407e76-93c5-40e9-b633-5c3922a96ab3
# ╠═3ca475e5-a56d-49f1-94db-b51faf923e0f
# ╠═cfe2b3e6-957a-4807-b4f9-80f9d465ba44
# ╟─f0c17f6a-2856-4de9-9981-d6d25a544e4f
# ╠═ea3775a0-9929-4b5a-ae06-2a343fe132f7
# ╟─82887c60-5d63-4d3f-bf4c-a89ee94bb1ad
# ╠═5b9514f6-acd8-4be7-9741-18e58c9377d7
# ╠═9fcaec3f-e442-4150-a759-2b78376b6f98
# ╠═fff9c3a4-673d-41b9-828e-1c46e527c51b
# ╠═e66e149b-4c0a-4ecb-b793-d270777ab1a2
# ╠═ce736d70-6e43-4595-bf90-bea0497e0655
# ╠═ec23667b-13c5-4ced-b218-215b846825fc
# ╠═e6e48303-ef1d-4e8d-84c9-077b693fd39b
# ╠═b565f3dd-6f3d-4a9f-9e25-7038ced1b378
# ╠═0c57e155-40e1-4e5c-9960-66c2fe1d7e86
# ╠═8a1b9517-5645-42e0-8e39-17f0d66ec7b1
# ╠═a061819a-391a-405b-b833-a398e3d279d1
# ╟─cfdae888-3226-4230-979a-b02dd5b31ecd
# ╟─14d4ed18-4fa5-4237-a57e-08014f4c589d
# ╠═afb1343d-06c6-4450-be00-fee1e6ca1f44
# ╠═d4d02d50-376c-4bec-a72f-35d5d57eda2c
# ╠═a28d1e12-84bd-404f-a502-3d8d6809b032
# ╠═97c1146f-4803-42f2-9ca2-5258435c28aa
# ╠═615d7bd8-e388-46eb-9860-deff32126b6b
# ╠═d347a153-ee3d-4265-aa24-7b0cda95fcb3
# ╟─105a6c92-c85f-4b71-9a5a-8cf5f0fc59f5
# ╠═9155e950-6608-415a-b932-78ff67e1b4bd
# ╠═58aa834c-db17-4274-829c-669d393fbbc6
# ╠═50e605ea-37b3-49ec-802f-9f7a7bf04a20
# ╠═53047746-8e91-4899-9441-2d2127ffe7f2
# ╠═6b077bbb-c8a0-4d06-b8d9-adff66d90fb2
# ╟─5b648ec8-d3fd-48cf-9ec4-083eb0363303
# ╠═165ccdf9-ae5b-42c9-b372-ef8676a07697
# ╠═d348d150-be97-43fc-a9ca-dc3e9e6960af
# ╠═1cba7665-1745-49e6-8fb6-72d6373b7d3f
# ╠═6048782e-3103-4e72-8a86-476e28b0a623
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
