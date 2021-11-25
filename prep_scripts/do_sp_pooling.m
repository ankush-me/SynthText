

function [one_feat_sps, weight_pool_info]=do_sp_pooling(one_feat_img, one_sp_info)


img_size=size(one_feat_img); 
num_units=img_size(1)*img_size(2);
dim=img_size(3);
one_feat_img=reshape(one_feat_img, [num_units dim]);
img_size_org=one_sp_info.img_size;

pixel_ind_map=reshape([1: num_units], [img_size(1) img_size(2)]);
pixel_ind_map_org=imresize(pixel_ind_map, img_size_org, 'nearest');

pixel_ind_sps=one_sp_info.pixel_ind_sps;
num_sp=numel(pixel_ind_sps);
weight_pool_info=zeros([num_sp, num_units], 'like', one_feat_img);


for idx_sp=1:num_sp
    
    pixel_ind_sp_one=pixel_ind_sps{idx_sp};
    ind_pixels_in_map=pixel_ind_map_org(pixel_ind_sp_one);
    
    [ind_units,~,uniqueIndex] = unique(ind_pixels_in_map);
    frequency = accumarray(uniqueIndex(:),1)./numel(ind_pixels_in_map);
    frequency=single(frequency);
    
    freq_one_sp=zeros(1, num_units, 'single');
    
    freq_one_sp(ind_units)=frequency;

    weight_pool_info(idx_sp, :)=freq_one_sp;    

    
end


one_feat_sps=weight_pool_info*one_feat_img;



end