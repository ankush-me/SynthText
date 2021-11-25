function depths_filled = do_inpainting(depths, img_data, sp_info)

    
    depths=double(depths);    
    
    centroid_mask=gen_sp_centroid(sp_info);
    
    img_data=im2double(img_data);
    one_depth_filled = my_fill_depth_colorization(img_data, depths,  centroid_mask);
    depths_filled=single(one_depth_filled);

    
end