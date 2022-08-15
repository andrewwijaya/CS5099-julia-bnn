#=
Implementation of AdvancedHMC.jl example for the neural network code. The neural network code here is a slightly modified version compared to what is in BayesVI.jl - this is in order to make it work with HMC sampling.
=#
module BayesHMC
    using AdvancedHMC, Distributions, ForwardDiff, Zygote
    using Random
    using OMEinsum
    using Plots
    using OrderedCollections
    
    #= 
    This function separates the means and log standard deviations in a 1xD parameters vector.
    The function simply splits the input vector into two, treats the first half as means, the second half as log standard deviations.
    Input: 1xD vector of floats
    Output: 1x(D/2) vector of means, 1x(D/2) vector of log standard deviations.
    =#
    function unpack_params(parameters)
        mean, log_std = parameters[1:Int(length(parameters)/2)], parameters[Int(length(parameters)/2) + 1:end]
        return mean, log_std
    end

    #= Function to specify, and produce the Bayesian Neural Network.
    Input:
        layer_sizes = vector of integers specifying number of nodes in each layer. Example: [1, 20, 20, 1]
        L2_reg = a single float value specifying the L2 regularisation.
        noise_variance = float value, used for shrinkage effect on likelihood term.
        nonlinearity = function used as non-linearity between layers.
    Output:
        num_weights = total number of weights in the produced neural network.
        predictions = forward pass function. 
        logprob = log probability function.
    =#
    function make_neural_network_functions(layer_sizes, L2_reg, noise_variance, nonlinearity)
        shapes = collect(zip(layer_sizes[1:end-1], layer_sizes[2:end]))
        num_weights = sum((m+1)*n for (m, n) in shapes)
        function unpack_layers(weights)
            W = OrderedDict()
            b = OrderedDict()
            i=1
            for (m, n) in shapes
                W[i] = reshape(weights[1:m*n], (m, n))
                b[i] = reshape(weights[m*n:(m*n+n)-1], (1, n))
                weights = weights[(m+1)*n:end]
                i += 1
            end
            return (W, b)
        end

        # Outputs the predictions for each number of models sampled from posterior.
        # inputs dimension: observations x features.
        # weights dimensions: 
        function predictions(weights, inputs)
            #inputs = reshape(inputs, 1, size(inputs)...)
            params = unpack_layers(weights)
            Ws = params[1]
            bs = params[2]
            #Go through all samples for each layer. (W,b) is the collection of all samples of weights and biases for a particular layer.
            for j in range(1, length(Ws))
                W = Ws[j]
                b = bs[j]
                outputs = ein"ij,jk->ik"(inputs, W) .+ b
                inputs = nonlinearity(outputs)
                if j == length(Ws)
                    return outputs
                end
            end
        end

        # This method returns a vector of length [mc_samples] containing the log probability value for each sample.
        # weights = mc_samples x num_weights (non-variational)
        # outputs = vector of likelihoods of size mc_samples
        function logprob(weights, inputs, targets)
            log_prior = -L2_reg * sum(weights.^2, dims=1)
            preds = predictions(weights, inputs)
            log_lik = -sum((preds .- targets).^2, dims=1)[:,1]./noise_variance
            return log_prior[1] + log_lik[1]
        end

        return num_weights, predictions, logprob
    end
    
    #=
    RBF activation function.
    =#
    function rbf(x)
        exp.(-x.^2)
    end
    
    #=
    Linear activation function.
    =#
    function linear(x)
        x
    end
    
    #=
    Executes HMC sampler on the bayesian network. This is mostly from the example in the AdvancedHMC.jl page, 
    adapted to work with the neural network's posterior instead.
    Inputs:
        n_samples: number of samples
        n_adapts: number of adapts
        num_weights: number of weights in the neural network
        log_posterior: log posterior of the neural network
    Outputs:
        samples: samples from the posterior
        stats: statistics from AdvancedHMC.jl sampling process
    =#
    function run_hmc_sampler(n_samples, n_adapts, num_weights, log_posterior)
        # Choose parameter dimensionality and initial parameter value
        D = num_weights
        initial_θ = rand(D)

        # Define the target distribution
        ℓπ(θ) = log_posterior(θ)

        # Configure and run HMC sampler.
        metric = DiagEuclideanMetric(D)
        hamiltonian = Hamiltonian(metric, ℓπ, Zygote)
        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)
        proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

        # Samples stores the samples, stats contains diagnostic statistics information.
        samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=false, drop_warmup=true);
        return samples, stats
    end

    export unpack_params
    export make_neural_network_functions
    export rbf
    export linear
    export run_hmc_sampler
end