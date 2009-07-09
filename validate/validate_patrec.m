clear all;
close all;

%Creating the data for validation.
nClasses = 2;
c1 = [randn(1,3000); randn(1,3000)];
c2 = [2.5 + randn(1,3000); 2.5 + randn(1,3000)];

%Creating the neural network.
net = newff2([nClasses,2,1], {'tansig', 'tansig'});
net.trainParam.epochs = 3000;
net.trainParam.max_fail = 50;
net.trainParam.show = 1;
net.trainParam.batchSize = 1000;
net.trainParam.useSP = true;


%Creating the training, validating and testing data sets.
inTrn = {c1(:,1:3:end) c2(:,1:3:end)};
inVal = {c1(:,2:3:end) c2(:,2:3:end)};
inTst = {c1(:,3:3:end) c2(:,3:3:end)};

tic
[net, evo] = ntrain(net, inTrn, inVal, inTst);
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
plot(evo.epoch, evo.mse_trn, 'b-', evo.epoch, evo.mse_val, 'r-', evo.epoch, evo.mse_tst, 'k-', evo.epoch, evo.sp_val, 'm-', evo.epoch, evo.sp_tst, 'g-');
legend('MSE (trn)', 'MSE (val)', 'MSE (tst)', 'SP (val)', 'SP (tst)', 'Location', 'Best');
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

