function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    % error('not yet implemented');
    num_config = size(visible_state, 2);
    
    % an activated edge is an edge with both ends turned on
    edge_activations = hidden_state * visible_state'; % sum of edge activations overall configurations
    weighted_edge_activations = edge_activations .* rbm_w; 
    G = sum(sum(weighted_edge_activations)) / num_config; % average goodness across configurations
end
