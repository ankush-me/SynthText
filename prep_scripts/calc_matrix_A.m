function [mat_A] = calc_matrix_A( betas, pws_info)

num_sps=size(pws_info{1}, 1);
mat_I =  eye(num_sps, num_sps);

mat_D_minus_R=zeros(num_sps, num_sps);
for idx_pws=1:numel(betas)
    mat_D_minus_R=mat_D_minus_R + full(pws_info{idx_pws}) * betas(idx_pws);
end

mat_A = mat_I + mat_D_minus_R;

end

