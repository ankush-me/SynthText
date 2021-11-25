function centroid_mask=gen_sp_centroid(sp_info)

  img_size=sp_info.img_size;
  centroid_mask=false(img_size);
  
  pixel_ind_sps=sp_info.pixel_ind_sps;
  sp_num=length(pixel_ind_sps);
  
  
   for sp_idx=1:sp_num
    
    pixel_inds=pixel_ind_sps{sp_idx};
    one_sp_mask=zeros(img_size);
    one_sp_mask(pixel_inds)=1;
    stats = regionprops(one_sp_mask,'Centroid');

    centroid = round(stats.Centroid);   

    assert( centroid(1) <= img_size(2));
    assert( centroid(2) <= img_size(1));
    
    centroid_mask(centroid(2), centroid(1)) = 1;
    
  end
  