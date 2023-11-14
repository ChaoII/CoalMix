c=double(solution)*coalInfo(:,3:end-1)/6;

%         disp(solution_int8);
%         gcd_matrix = ones(1, m, 'uint8');
%         for i =1 :m
%             tmp = solution_int8(i,solution_int8(i,:)~=0);
%             if length(tmp)==2
%                 gcd_i = gcd(tmp(1), tmp(2));
%             elseif length(tmp)==1
%                 gcd_i = gcd(tmp(1), 0);
%             elseif isempty(tmp)
%                 gcd_i = gcd(0,0);
%             else
%                 error('每个原煤仓最多支持两种煤上仓');
%             end
%             gcd_matrix(1,i) = gcd_i;
%         end  
%         % 混煤比例
%         mix_case = solution_int8./gcd_matrix';
%         disp(mix_case);