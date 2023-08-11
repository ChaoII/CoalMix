function [mix_case,mix_info,mix_price] = getBestCase(coalInfo, unitConstraint, containerConstraint, feederCapacity, mixRatio, mutexCoal, standCoalQty, maxMixCoal)   
    %%
    % 功能：由给定备选上仓煤种，在满足机组约束、煤仓约束和比例约束条件下，生成若干最优配仓方案（原煤均价最低）;
    % 输入：以给定备选 n 种煤，配煤比例 k 种，机组共 m 个仓，生成top s 个最优解为例：
    %       coalInfo，可供上仓的原煤信息，字段：[序号，煤量，煤质(3~8)，原煤价]
    %          -----< n x 9 > 矩阵；
    %       unitConstraint，机组约束信息，字段：[热值、硫分、灰分、挥发分、全水、灰熔点]，行排列
    %          -----< 6 x 2 > 矩阵；
    %       containerConstraint，煤仓约束信息，字段：[启用，固定挥发分，煤1，煤2，比1，比2，煤质约束(7~18)]
    %          -----< m x 18 > 矩阵；
    %       feederCapacity，运行给煤机最大出力 x 计划时段时长，如：feederCapacity = 75(t/h) x 8 (h) = 600 (t)
    %          -----< 1 x 1 > 矩阵（标量）；
    %       mixRatio，配煤比例，（注：类似 1:2 和 2:1 算同一种比例）
    %          -----< k x 2 > 矩阵，如[0 1; 1 1]、[0 1; 1 1; 1 2]、[0 1; 1 1; 1 2 ;1 3],etc.
    %       mutexCoal，互斥煤种，两种煤只能取其中一种，每一行代表互斥的两种煤，数量不多于Cn2
    %          -----< t x 2 > 矩阵，如[0 1; 0 2; 0 3; 1 3; 1 5; 3 7; 4 8]etc.
    %       standCoalQty，标煤量
    %          -----< k x 2 > 矩阵，如[0 1; 1 1]、[0 1; 1 1; 1 2]、[0 1; 1 1; 1 2 ;1 3],etc.
    %       maxMixCoal，最大配煤种数，配煤方案中，最多有当前种煤种，太多可操作性不强
    %          -----< k x 2 > 矩阵，如[0 1; 1 1]、[0 1; 1 1; 1 2]、[0 1; 1 1; 1 2 ;1 3],etc.
    % 输出：输出 top s 的配仓方案和掺配结果 ，
    %       mixScheme，top s（本方法取s=1） 的配仓方案，如果煤仓上单种煤则为 1 ，否则按其原比例输出，如1:2、1:3
    %          -----< (s*m) x n > 矩阵；
    %       result，对应配仓方案的机组各平均值：[热值, 硫分, 灰分, 挥发分, 全水, 灰熔点, 原煤均价]，
    %          -----< s x 7 > 矩阵；
    % 说明：（1）该算法为通过MATLAB YALMIP优化算法工具箱前端调用IBM ILOG Cplex后端求解器，进行整数规划计算，需要目标电脑上安装有mcr和cplex。
    %       （2）当前使用的cplex为商业版，对于变量数量无1000个的最大限制
    %       变量个数计算公式为：nVar = nCoal*[nContainer*(n_b+1)+1] =15*[6*(n_b+1)+1]，
    %       n_b为辅助变量因子，个数与比例mixRatio的个数的一半接近，如比例[0 1; 1 1; 12]对应的n_b=1，
    %       比例[0 1; 1 1; 1 2 ;1 3]对应的n_b=2，由此可估计在6个煤仓情况下，加仓煤种最多可支持50个左右
    %       （3）求解强度、返回解数量、求解时间、在无解情况下确定矛盾约束等定制化需求都可以通过cplex类实现，具体用法请阅读cplex
    %       api。
    %       （4）在编译为dll时，请确保mcr版本、cplex版本、c++编译器版本及编译平台位数一致，否则无法被调用。
    %% ------------------------
    save("mock10.mat","coalInfo", "unitConstraint", "containerConstraint", "feederCapacity", "mixRatio", "mutexCoal", "standCoalQty", "maxMixCoal");
    display(coalInfo);
    display(unitConstraint);
    display(containerConstraint);
    display(feederCapacity);
    display(mixRatio);
    display(mutexCoal);
    display(standCoalQty);
    display(maxMixCoal);
    mixRatio = mixRatio./(gcd(mixRatio(:,1),mixRatio(:,2))*ones(1,2));
    sum_mixRatio = sum(mixRatio, 2);
    slcm = 1;
    for i = 1: length(sum_mixRatio)
        slcm = lcm(slcm, sum_mixRatio(i));
    end
    ele_m = mixRatio./repmat(sum_mixRatio, 1, 2)*slcm;
    ele = unique(ele_m);
    
    % 煤仓数量
    m = size(containerConstraint,1);
    % 煤种数量
    n = size(coalInfo, 1);
    x = intvar(m, n);
    % 总煤量 煤仓数 * 单仓量(slcm个单位)
    total_quality = slcm* m;
    % 约束0煤的数量不能为负
    constrat0 = x >= zeros(m, n);
    % 约束1煤仓给煤机出力约束 所有仓同一时间煤量相同，都是12个单位
    constrat1 = sum(x, 2)==slcm.*ones(m, 1);
    % 约束2最多允许两种煤上仓，也就是每一行中非0的数小于等于2
    constrat2 = [];
    for i=1:m
        constrat2= [constrat2,nnz(x(i,:))<=2];
    end
    % 约束3，必须是在允许的比例之间[1:1,1:2,1:3,0:1]
    constrat3 = ismember(x, ele');
    % 约束4煤量约束sum(x,1) 每一种煤的总量
    constrat4 = sum(x,1)'/slcm <= coalInfo(:,2)/feederCapacity;
    % 约束5锅炉煤质约束 sum(x,1)即为所有煤仓中煤种的量，均匀混合所有煤仓的煤后的媒质
    temp = (sum(x,1)*coalInfo(:,3:end-1)/total_quality)';
    constrat5 = (unitConstraint(:,1)<=temp) & (temp<=unitConstraint(:,2));
    % 约束6 煤仓煤质约束(单仓12个煤量单位)
    lb_index = 7:2:17;
    ub_index = lb_index+1;
    temp = x*coalInfo(:,3:end-1)/slcm;
    constrat6 = (containerConstraint(:,lb_index) <= temp) & (temp <= containerConstraint(:,ub_index));
    % 约束7 煤仓固定挥发分约束
    constrat7=[];
    % 约束8 煤仓固定比例约束
    container_coal = containerConstraint(:,3:6);
    [r,~] = find(sum(container_coal,2)~=0);
    constrat8 = x(container_coal(r,[1,2]))==container_coal(r,[3,4])*slcm;
    % 约束9 互斥性约束，在所有煤种中有a不能有b
    constrat9=[];
    for i=1:length(mutexCoal)
        size_len = length(mutexCoal{i});
        for j = 1:m
            constrat9 =[constrat9, sum(repmat(x(j,:),size_len,1)-repmat(mutexCoal{i}',1,n),'all')<=1];
        end
    end

    % 约束10 热值守恒
    constrat10=[];
%     constrat10 = abs((sum(x,1)*coalInfo(:,3)/total_quality)'*feederCapacity-200000)<=standCoalQty*7000;
%     constrat10 = standCoalQty*7000 == (sum(x,1)*coalInfo(:,3)/total_quality)'*feederCapacity;
    % 约束11 最大煤种约束
    constrat11=[];
    constrat11 = nnz(sum(x,1))<= maxMixCoal; 
    %目标函数(1XN)x(1XN)
    obj = sum(x,1)*coalInfo(:,end);
    Constraints = [constrat0, constrat1, constrat2, constrat3, constrat4, constrat5, constrat6, constrat7, constrat8, constrat9, constrat10, constrat11];
    % 创建优化模型
    ops = sdpsettings('solver','cutsdp', 'verbose', 0,'savesolveroutput', 2, 'savesolverinput', 1,'allownonconvex', 1); % 选择整数规划求解器
    ops.cutsdp.SolCount = 3 ;
    res= optimize(Constraints, obj, ops);
    display(res);
    display(res.info);
    if(res.problem==0)
        % 获取解
        solution = uint8(value(x));
        gcd_matrix = ones(1,m,'uint8');
        for i =1 :m
            tmp = solution(i,solution(i,:)~=0);
            if length(tmp)==2
                gcd_i = gcd(tmp(1),tmp(2));
            elseif length(tmp)==1
                gcd_i = gcd(tmp(1),0);
            elseif isempty(tmp)
                gcd_i = gcd(0,0);
            else
                error('每个原煤仓最多支持两种煤上仓');
            end
            gcd_matrix(1,i) = gcd_i;
        end  
        % 混煤比例
        mix_case = solution./gcd_matrix';
        disp(mix_case);
        % 混煤媒质
        mix_info = sum(solution,1)*coalInfo(:,3:end-1)/total_quality;
        disp(mix_info);
        % 最少煤价
        mix_price = sum(solution,1)*coalInfo(:,end)/total_quality;
        disp(mix_price);
    else
        disp("求解失败！")
    end
    % 混煤媒质
end
