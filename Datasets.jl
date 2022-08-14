#=
Datasets.jl contains three synthetic regression datasets used for testing of HMC and VI Bayesian Neural Networks.
=#
module Datasets
    function build_dataset_tfp(n=150, n_tst=150)
        w0 = 0.125
        b0 = 5.
        x_range = [-20, 60]
        function s(x)
            g = (x .- x_range[1]) / (x_range[2] - x_range[1])
            return 3 .* (0.25 .+ g.^2.)
        end
        x = (x_range[2] - x_range[1]) * rand(n) .+ x_range[1]
        eps = randn(n) .* s(x)
        y = (w0 .* x .* (1. .+ sin.(x)) .+ b0) .+ eps
        x_tst = collect(range(x_range[1], stop=x_range[2], length=n_tst))
        return y, x, x_tst
    end

    # Synthetic regression dataset as provided in the Autograd blackbox svi in five lines of python paper.
    function build_dataset_1(n_data=40, noise_std=0.1)
        D=1
        inputs = vcat(LinRange(0,2,Int(n_data/2)), LinRange(6,8,Int(n_data/2)))
        targets = cos.(inputs) .+ randn(n_data) .* noise_std
        inputs = (inputs .- 4) ./4
        inputs = reshape(inputs, (length(inputs),D))
        targets = reshape(targets, (length(targets),D))
        return inputs, targets
    end
    
    # Same as dataset 1 but a more expanded version.
    function build_dataset_2(n_data=40, noise_std=0.1)
        D=1
        inputs = LinRange(0,32,Int(n_data))
        targets = cos.(inputs) .+ randn(n_data) .* noise_std
        inputs = (inputs .- 4) ./4
        inputs = reshape(inputs, (length(inputs),D))
        targets = reshape(targets, (length(targets),D))
        return inputs, targets
    end
    
    # Dataset from https://ekamperi.github.io/machine%20learning/2021/01/07/probabilistic-regression-with-tensorflow.html#tensorflow-example
    function build_dataset_3()
        n_points = 100
        x_train = collect(LinRange(-1, 1, n_points))
        y_train = x_train.^5 + 0.4 .* x_train .* randn(n_points)
        return reshape(x_train, (size(x_train)...,1)), reshape(y_train, (size(y_train)...,1))
    end

    export build_dataset_tfp
    export build_dataset_1
    export build_dataset_2
    export build_dataset_3
end