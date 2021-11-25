function depths_pred  = do_model_evaluate(model, ds_info, opts)


img_data=ds_info.img_data;
sp_info=ds_info.sp_info;
net=model.net;
label_norm_info=model.label_norm_info;
img_norm_info=model.image_norm_info;

if opts.useGpu
    img_data=gpuArray(img_data) ;
    net = vl_simplenn_move(net, 'gpu');
end


image_mean=img_norm_info*ones(size(img_data), 'single');
img_data=single(img_data);
img_data = img_data - image_mean;
ds_info.img_data=img_data;    
    
    
fprintf('network forward for depth prediction...\n');
res = my_nn_struct_forward(net, img_data, [], ...
    'disableDropout', true, ...
    'conserveMemory', true, ...
    'sync', true, ...
    'ds_info', ds_info, 'keep_layer_output', false) ;


output_info=squeeze(gather(res(end).output_info));
if isfield(output_info, 'predict_labels')
    output_info=output_info.predict_labels;
end
depths_sps=output_info(:);


depths_sps=project_back_label(depths_sps, label_norm_info);

depths_pred=depths_sps(sp_info{1}.sp_ind_map);




end




