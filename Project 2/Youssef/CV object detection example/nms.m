function local_peaks = nms(T_, sumThres)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DO NOT MODIFY THIS FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('starting of nmx ...\n');
tempScoreMatrix = T_;
[max_row_ind, max_col_ind] = size(T_);
topN_Index = [];
topN_Score = [];
J_Mask_Cell = cell(0);

while(true)
    [max_per_row, col_No_vec] = max(tempScoreMatrix,[],2);
    [max_max, row_No] = max(max_per_row);
    col_No = col_No_vec(row_No);
     
    if (max_max) > 0
        B_ = tempScoreMatrix > 0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('   starting of region_growing ...\n');
        J = region_growing(B_,row_No,col_No);
        fprintf('   ending of region_growing ...\n');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        J_vect = J(:);
        J_one = J_vect==1;
        tempScoreMatrix(J_one) = -inf;    
        if sum(sum(J)) > sumThres
            topN_Index = [topN_Index; row_No, col_No];
            topN_Score = [topN_Score; T_(row_No,col_No)];
            J_Mask_Cell{end+1} = J;
        end
    else 
        break;
    end
end

%% output
local_peaks.topN_Index = topN_Index;
local_peaks.topN_Score = topN_Score;
local_peaks.J_Mask_Cell = J_Mask_Cell;
fprintf('ending of nmx ...\n');
end