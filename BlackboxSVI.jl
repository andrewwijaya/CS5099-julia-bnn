module BlackboxSVI
    using Distributions
    #=
    Implementation of blackbox stochastic variational inference. This function returns the variational objective given the log
    posterior.
    Input:
        logprob = log posterior.
        D = number of parameters.
        mc_samples = number of MCMC samples used to approximate the gradient.
    Output:
        variational_objective = variational log posterior, the gradient of which can be used for gradient descent optimisation.
    =#
    function black_box_variational_inference(logprob, D, mc_samples)        
        #= 
        Vectorised calculation of Gaussian entropy.
        Input: vector of log standard deviations.
        Output: vector of Gaussian entropies.
        =#
        #TODO: is there something wrong with this calculation for D>1?
        function gaussian_entropy(log_std)
           return 0.5 * D * (1.0 + log(2*π)) + sum(log_std)
        end

        #= 
        Evaluates Evidence LOwer Bound (ELBO) of the parameters.
        Input: vector of parameters containing means and standard deviations of each weight in the neural network.
        Output: a single ELBO value.
        =#
        function variational_objective(params)
            mean_vals, log_std = unpack_params(params)
            # Reparameterisation trick
            samples = randn(mc_samples, D) .* exp.(log_std)' .+ mean_vals'
            lower_bound = gaussian_entropy(log_std) + mean(logprob(samples))
            return -lower_bound
        end
        
        return variational_objective, unpack_params
    end

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

    export black_box_variational_inference
    export unpack_params
end