#=
Implementation of Variational Inference Bayesian Neural Networks. This implementation is mostly a port of the Python implementation from Autograd. A custom Neural Network is implemented, and learning is done using Flux.jl. The variational inference implemented uses both the mean field and fixed Gaussian form assumptions.
=#
module BayesVI
    using Flux
    using Distributions
    using Plots
    using OrderedCollections
    using Random
    using OMEinsum
    using LinearAlgebra
    using BenchmarkTools
    include("./BlackboxSVI.jl")
    using .BlackboxSVI

    #= Function to specify, and produce the Bayesian Neural Network.
    Input:
        layer_sizes = vector of integers specifying number of nodes in each layer. Example: [1, 20, 20, 1]
        L2_reg = a single float value specifying the L2 regularisation.
        noise_variance = TODO
        nonlinearity = function used as non-linearity between layers.
        [TODO] add variable for final nonlinearity to add flexibility for final layer.
    Output:
        num_weights = total number of weights in the produced neural network.
        predictions = forward pass function. 
        logprob = log probability function.
    =#
    function make_neural_network_functions(layer_sizes, L2_reg, noise_variance, nonlinearity)
        shapes = collect(zip(layer_sizes[1:end-1], layer_sizes[2:end]))
        num_weights = sum((m+1)*n for (m, n) in shapes)
        function unpack_layers(weights)
            num_weights_sets = size(weights)[1]
            W = OrderedDict()
            b = OrderedDict()
            i=1
            for (m, n) in shapes
                W[i] = reshape(weights[:,1:m*n], (num_weights_sets, m, n))
                b[i] = reshape(weights[:,m*n:(m*n+n)-1], (num_weights_sets, 1, n))
                weights = weights[:, (m+1)*n:end]
                i += 1
            end
            return (W, b)
        end

        # Outputs the predictions for each number of models sampled from posterior.
        # inputs dimension: observations x features
        # weights dimensions: 
        function predictions(weights, inputs)
            inputs = reshape(inputs, 1, size(inputs)...)
            params = unpack_layers(weights)
            Ws = params[1]
            bs = params[2]
            
            inputs_stacked = vcat()
            for i in range(1, size(weights)[1])
                inputs_stacked = vcat(inputs_stacked, inputs)
            end
            inputs = inputs_stacked

            #Go through all samples for each layer. (W,b) is the collection of all samples of weights and biases for a particular layer.
            for j in range(1, length(Ws))
                W = Ws[j]
                b = bs[j]
                outputs = ein"mnd,mdo->mno"(inputs, W) .+ b
                inputs = nonlinearity(outputs)
                if j == length(Ws)
                    return outputs
                end
            end
        end
        
        function ensemble_prediction(params, inputs, number_of_ensemble)
            mean_vals, log_std = unpack_params(params)
            samples = randn(number_of_ensemble, length(mean_vals)) .* exp.(log_std)' .+ mean_vals'
            mean(predictions(samples, inputs), dims=1)
        end

        # This method returns a vector of length [mc_samples] containing the log probability value for each sample.
        # weights = mc_samples x num_weights (non-variational)
        # outputs = vector of likelihoods of size mc_samples
        function logprob(weights, inputs, targets)
            log_prior = -L2_reg * sum(weights.^2, dims=2)
            preds = predictions(weights, inputs)
            log_lik = -sum((preds .- targets').^2, dims=2)[:,1]./noise_variance
            return log_prior + log_lik
        end

        return num_weights, predictions, logprob, ensemble_prediction
    end

    # Create samples of weights based on variational parameters.
    function sample_posteriors(variational_parameters)
        means, log_stds = unpack_params(variational_parameters)
        #length(means) is equivalent to the number of weights in the (non-variational) neural network
        return randn(1, length(means)) .* exp.(log_stds)' .+ means'
    end
    
    # Initialise variational parameters randomly.
    function initialise_variational_parameters(num_weights)
        init_mean = randn(num_weights)
        init_log_std = -5 * ones(num_weights)
        return vcat(init_mean, init_log_std)
    end
    
    function rbf(x)
        exp.(-x.^2)
    end
    
    function linear(x)
        x
    end
    
    function train_bayesian_neural_network(epochs, learning_rate, num_weights, objective)
        init_var_params = initialise_variational_parameters(num_weights)
        param_hist = Array{}[]
        opt = ADAM(learning_rate)
        elbos = zeros(epochs)
        for i in range(1,epochs)
            Flux.Optimise.update!(opt, init_var_params, gradient(objective, (init_var_params))[1])
            push!(param_hist, copy(init_var_params))
            elbos[i] = -objective(init_var_params)
        end
        return param_hist, elbos
    end
    
    # Sample posteriors and plot predictions.
    function sample_and_plot(inputs, targets, init_var_params, number_of_models, prediction_fn, title_name="", ylims=(-3, 3), xlims=(-8, 8), xmin=-8, xmax=8, xrange=150)
        plot_inputs = collect(LinRange(xmin, xmax, xrange))
        plot_inputs = reshape(plot_inputs, (length(plot_inputs), 1))
        scatter(inputs, targets, label="")
        for i in range(1, number_of_models)
            sample_weights = sample_posteriors(init_var_params)
            outs = prediction_fn(sample_weights, plot_inputs)
            plot!(plot_inputs, outs[1, :, 1], ylim=ylims, xlim=xlims, size=(700,400), label="", title=title_name)
        end
    end

    function animate_variational_params(inputs, targets, param_hist, number_of_models, prediction_fn, ylims=(-3, 3), xlims=(-8, 8), xmin=-8, xmax=8, xrange=150)
        epochs = length(param_hist)
        anim = @animate for i in range(1, epochs)
            title="Iteration: "*string(i)*"/$epochs - Bayesian Neural Network"
            sample_and_plot(inputs, targets, param_hist[i], number_of_models, prediction_fn, title, ylims, xlims, xmin, xmax, xrange)
        end
#         return anim
        gif(anim, fps=20)
    end

    export unpack_params
    export black_box_variational_inference
    export make_neural_network_functions
    export sample_posteriors
    export initialise_variational_parameters
    export sample_and_plot
    export rbf
    export linear
    export train_bayesian_neural_network
    export animate_variational_params
end