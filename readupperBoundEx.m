upperBoundList = [0.001 0.004 0.007 0.01 .011 0.012 0.013 0.016]';
% information matrix has two columns.
% first column would indicate proportion of experiments in which 
% number of agents between .4 and .6 is more than 18 agents.
% 
% Second column would indicate the case in which number of agenst
% at extremes are more than 57.
information = zeros(length(upperBoundList),3);

for upBoundCount = 1:length(upperBoundList) 
    for initCount = 1:20
        fileName = sprintf('%s%s%s%s%s.mat','uB', ...
                           num2str(upperBoundList(upBoundCount)), ...
                           '*uS0.0','*initCount',num2str(initCount));
        data = load(fileName);
        Ex_names = fieldnames(data);
        Ex_names = sort(Ex_names);
        for exCount = 1:25
            information(upBoundCount,1) = information(upBoundCount,1) + ...
                                          sum(data.(Ex_names{exCount})(end,:) <= 0.2);
            
            information(upBoundCount,2) = information(upBoundCount,2) + ...
                                          sum(data.(Ex_names{exCount})(end,:) >= 0.4 & ...
                                          data.(Ex_names{exCount})(end,:) <= 0.6);
                                                  
            information(upBoundCount,3) = information(upBoundCount,3) + ...
                                          sum(data.(Ex_names{exCount})(end,:) >= 0.8);
        end
    end
end