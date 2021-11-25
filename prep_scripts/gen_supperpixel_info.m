



function sp_info=gen_supperpixel_info(img_data, sp_size)


sp_info=do_gen_sp_info(img_data, sp_size);

sp_info.relation_infos=do_gen_relation_info(sp_info);



end







function sp_info=do_gen_sp_info(img_data, sp_size)


img_data=im2single(img_data);

seg_map = vl_slic(img_data, sp_size, 0.1, 'MinRegionSize', 10) ;

[sp_inds, ~, tmp_seg_map]=unique(seg_map);
seg_map=uint16(seg_map);
seg_map(:)=tmp_seg_map;
sp_num=length(sp_inds);
assert(sp_num<2^16);


pixel_ind_sps=cell(sp_num, 1);
for sp_idx=1:sp_num
    one_pixel_inds=find(seg_map==sp_idx);
    pixel_ind_sps{sp_idx}=uint32(one_pixel_inds);
end


sp_info=[];
sp_info.sp_ind_map=seg_map;
sp_info.img_size=size(seg_map);
sp_info.sp_num=sp_num;
sp_info.pixel_ind_sps=pixel_ind_sps;



end






function relation_infos=do_gen_relation_info(sp_info)


  map=sp_info.sp_ind_map;
  pixel_ind_sps=sp_info.pixel_ind_sps;
  
  sp_num=length(pixel_ind_sps);
  relation_infos=cell(sp_num, 1);
  
  map1 = circshift(map, [1 0]);
  map1(1,:) = map(1,:);
  map2 = circshift(map, [-1 0]);
  map2(end,:) = map(end,:);
  map3 = circshift(map, [0 1]);
  map3(:,1) = map(:,1);
  map4 = circshift(map, [0 -1]);
  map4(:,end) = map(:,end);

  adjacent_mat=false(sp_num, sp_num);
  
  for sp_idx=1:sp_num
      
    one_sp_info=[];
    
    ind = pixel_ind_sps{sp_idx};
    adj = [map1(ind) map2(ind) map3(ind) map4(ind)];
    adj = unique(adj(:));
    adj = setdiff(adj, sp_idx);
        
    one_sp_info.adjacent_sp_inds = adj;
    
    
    adjacent_mat(sp_idx, adj)=true;
    adjacent_mat(adj, sp_idx)=true;
    adjacent_mat(sp_idx, sp_idx)=false;
    
    relation_infos{sp_idx}=one_sp_info;
  end
    
  adjacent_mat=adjacent_mat|adjacent_mat';
  
  for sp_idx=1:sp_num
      one_sp_info=relation_infos{sp_idx};
      adj_sp_inds=find(adjacent_mat(:, sp_idx));
      one_sp_info.adjacent_sp_inds=uint16(adj_sp_inds);
      relation_infos{sp_idx}=one_sp_info;
  end
  
  

end








