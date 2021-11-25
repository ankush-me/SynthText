function do_save_prediction( depth_inpaint, opts)


do_show_log_scale=opts.do_show_log_scale;


label_norm_info=opts.label_norm_info;
max_d=label_norm_info.max_d;
min_d=label_norm_info.min_d;
norm_type=label_norm_info.norm_type;
if  norm_type==2
    max_d=power(2, max_d);
    min_d=power(2, min_d);
elseif norm_type==3
    max_d=power(10, max_d);
    min_d=power(10, min_d);
end

if do_show_log_scale
    scaling_label=log10(max_d)-log10(min_d);
    offset_label=log10(min_d);
else
    scaling_label=max_d-min_d;
    offset_label=min_d;
end
    
   
depth_inpaint_show=depth_inpaint;
if do_show_log_scale
    depth_inpaint_show=log10(depth_inpaint); 
end
depth_inpaint_show=(depth_inpaint_show - offset_label)/scaling_label;
   
    
img_file_eval=opts.img_file_name;
    


img_save_dir = opts.result_dir;
  

[~, org_file_name]=fileparts(img_file_eval);
one_save_dir=fullfile(img_save_dir,org_file_name);

if ~exist(one_save_dir, 'dir')
    mkdir(one_save_dir);
end


one_cache_file='predict_depth_gray.png';
imwrite(depth_inpaint_show, fullfile(one_save_dir,one_cache_file));

one_cache_file='predict_depth_rgb.png';
depth_show=depth_inpaint_show;
depth_show=(depth_show-min(depth_show(:)))/(max(depth_show(:)) - min(depth_show(:)));
depth_show=depth_show*(64-1)+1;
depth_show=round(depth_show);
imwrite(depth_show, colormap('jet'), fullfile(one_save_dir,one_cache_file));


one_cache_file='predict_depth.mat';
data_obj=single(depth_inpaint);
save( fullfile(one_save_dir,one_cache_file), 'data_obj');




end


