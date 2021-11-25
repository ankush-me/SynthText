function label_data_trans=project_back_label(label_data, norm_info)


if isempty(norm_info)
    label_data_trans=label_data;
else
    scaling=norm_info.max_d-norm_info.min_d;
    offset=norm_info.min_d;

    if ~iscell(label_data)
        label_data=max(0, label_data);
        label_data=min(1, label_data);


        label_data_trans = label_data*scaling + offset;

        if  norm_info.norm_type==2
            label_data_trans=power(2, label_data_trans);    
        elseif norm_info.norm_type==3
            label_data_trans=power(10, label_data_trans);
        end

    else   
        num=numel(label_data);
        label_data_trans=cell(num, 1);
        for idx=1:num
            label_data_img=max(0, label_data{idx});
            label_data_img = min(1, label_data_img);
            label_data_trans{idx} = label_data_img*scaling + offset;

            if  norm_info.norm_type==2
                label_data_trans{idx}=power(2, label_data_trans{idx});    
            elseif norm_info.norm_type==3
                label_data_trans{idx}=power(10, label_data_trans{idx});
            end
        end    
    end

end

end