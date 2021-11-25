

function output_info=my_struct_obj_dcnf(X, layer_info, batch_ds_info)



img_num=length(batch_ds_info.img_idxes);
sp_num_imgs=batch_ds_info.sp_num_imgs;
assert(img_num==length(sp_num_imgs));


pws_info=batch_ds_info.pws_info;


betas=layer_info.struct_model_w;
betas=gather(betas);
X=gather(X);


predict_labels=zeros([size(X, 1)*img_num 1], 'single');

for idx_img=1:img_num
       
    idx_sp_batch_begin = sum(sp_num_imgs(1: max(0, idx_img-1)))+1; 
    idx_sp_batch_end = idx_sp_batch_begin+sp_num_imgs(idx_img)-1;
    
    
    xq = squeeze(X(:, :, :, idx_sp_batch_begin:idx_sp_batch_end));    
    z=xq;
    
    one_pws_info = pws_info(idx_img, :);
    mat_A = calc_matrix_A( betas, one_pws_info);

    y_inf = mat_A\z;
    
    predict_labels(idx_sp_batch_begin:idx_sp_batch_end) = y_inf;
    
     
end



output_info=[];
output_info.predict_labels=predict_labels;
output_info.objective_value=[];


end