warning off all
clear all;
close all;

%Creating the data for validation.
nClasses = 2;
%Creating the training, validating and testing data sets.
inTrn = {randn(2,3000), (2.5 + randn(2,5000))};
inVal = {randn(2,2500), (2.5 + randn(2,4500))};
inTst = {randn(2,2000), (2.5 + randn(2,4000))};

%Creating the neural network.
net = newff2(inTrn, [-1 1], 3, {'tansig', 'tansig'});
net.trainParam.epochs = 3000;
net.trainParam.max_fail = 50;
net.trainParam.show = 1;
net.trainParam.batchSize = 1000;
net.trainParam.useSP = true;

tic
[net, evo] = ntrain(net, inTrn, inVal);
toc

%Generating the testing outputs.
out = nsim(net, inTst);

%First analysis: RoC.
[sp, cut, det, fa] = genROC(out{1}, out{2}, 200);
[maxSP maxSP_idx] = max(sp);
 
figure;
plot(fa, det);
hold on;
plot(fa(maxSP_idx), det(maxSP_idx), 'bo');
title('RoC');
xlabel('False Alarm (%)');
ylabel('Detection (%)');
grid on;

figure;
plot(evo.epoch, evo.mse_trn, 'b-', evo.epoch, evo.mse_val, 'r-', evo.epoch, evo.sp_val, 'm-');
legend('MSE (trn)', 'MSE (val)', 'SP (val)', 'Location', 'East');
title('Training Evolution');
xlabel('Epoch');
ylabel('Performance Value');
grid on;

%Second analysis: output distribution.
figure;
histLog(out, 200);
title('Output Distribution for the FastNet (Cont) Version')
xlabel('Network Output');
ylabel('Counts');

