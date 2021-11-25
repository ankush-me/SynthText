

function [box_img_data, box_pos]=gen_box_img_pad(img_data_pad, one_sp_mask, box_size)


    [rows cols]=find(one_sp_mask);
    one_box=[min(rows) min(cols) max(rows) max(cols)];
    one_center=[one_box(1)+one_box(3) one_box(2)+one_box(4)];
    one_center=one_center./2;
    one_box2_point=one_center-box_size./2;


    % padding translation:
    one_box2_point=one_box2_point+box_size;
    one_box2=[one_box2_point(1) one_box2_point(2) ...
        one_box2_point(1)+box_size(1)-1 one_box2_point(2)+box_size(2)-1];
    one_box2=ceil(one_box2);

    box_img_data=img_data_pad([one_box2(1):one_box2(3)], [one_box2(2):one_box2(4)],:);
    
    %add by fayao
    box_pos=one_box2; %[YMIN, XMIN, YMAX, XMAX];

end





