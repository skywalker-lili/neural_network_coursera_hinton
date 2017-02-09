function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    
    % Turn continues visible_data each between 0-1 into a binary state vector
    % so that the RBM can use
    visible_data = sample_bernoulli(visible_data);
    
    % Get the 0-degree hidden state from visible data (0-degree visible state)
    hid_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hid_state_0 = sample_bernoulli(hid_prob);
    
    % Get the 1st-degree reconstruction from hidden state
    vis_prob = hidden_state_to_visible_probabilities(rbm_w, hid_state_0);
    vis_state_1 = sample_bernoulli(vis_prob);
    
    % Get the 1st-degree hidden state from 1st-degree reconstruction
    hid_prob = visible_state_to_hidden_probabilities(rbm_w, vis_state_1);
    % hid_state_1 = sample_bernoulli(hid_prob); % hid_state_1 isn't necessary because we only need the expected value of edge activations, which isn't changed by sample_bernoulli();
    
    % Caculate the grdient (expected value of edge activations) from visible data (0-degree visible state)
    %                                                           from reconstruction (1st-degree visible state)
    grad_0 = configuration_goodness_gradient(visible_data, hid_state_0);
    % grad_1 = configuration_goodness_gradient(vis_state_1, hid_state_1);
    grad_1 = configuration_goodness_gradient(vis_state_1, hid_prob);
    
    ret = grad_0 - grad_1;
end
