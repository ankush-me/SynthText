
function Y = my_nn_logistic(X,dzdy)


sX=size(X);

gpuMode=isa(X, 'gpuArray');

if gpuMode
    X_minus = minus(gpuArray.zeros(sX), X);
    mat_ones = gpuArray.ones(sX);
else
    X_minus = -X;
    mat_ones = 1;
end

exp_value=exp(X_minus);
logit_value=mat_ones./(mat_ones+exp_value);


if nargin <= 1
  
    Y=logit_value;
  
else
    
   Y = logit_value.*(mat_ones-logit_value);
   Y = Y .* dzdy ;
end


end
