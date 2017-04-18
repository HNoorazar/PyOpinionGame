% Here we take average of behavior of each individual

upperBoundList= (0.001:0.003:.016)';
upperBoundList = [0.001 0.004 0.007 0.01 .011 0.012 0.013 0.016]';
comNo = length(upperBoundList);
popSize = 75;
Behavioraverages = zeros(popSize, comNo);

for upBoundCount = 1:length(upperBoundList) 
    for initCount = 1:20
        fileName = sprintf('%s%s%s%s%s.mat','uB', ...
                           num2str(upperBoundList(upBoundCount)), ...
                           '*uS0.0','*initCount',num2str(initCount));
        data = load(fileName);
        Ex_names = fieldnames(data);
        Ex_names = sort(Ex_names);
        for exCount = 1:25
            Behavioraverages(:, upBoundCount) = Behavioraverages(:, upBoundCount) + ...
                                                data.(Ex_names{exCount})(end,:)';
        end
    end
    Behavioraverages(:, upBoundCount) = Behavioraverages(:, upBoundCount)/(20*25);
end