
function [layer_output_info, gradient_info]= my_nn_structured_loss(X, layer_info, ds_info, dzdy)


if nargin<=3
    
    layer_output_info=layer_info.objective_fn(X, layer_info, ds_info);
    
else

    layer_output_info=[];
    gradient_info=layer_info.gradient_fn(X, layer_info, ds_info, dzdy);
      
    
    
end




end







