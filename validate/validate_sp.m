clear all;
close all;

%Creating the data for validation.
nClasses = 2;
Nev = 12000;
c1 = [randn(1,Nev); randn(1,Nev)];
c2 = [2.5 + randn(1,Nev); 2.5 + randn(1,Nev)];

%Creating the training, validating and testing data sets.
inTrn = {c1(:,1:3:end) c2(:,1:3:end)};
inVal = {c1(:,2:3:end) c2(:,2:3:end)};
inTst = {c1(:,3:3:end) c2(:,3:3:end)};

%Creating the neural network.
inNet = newff2(inTrn, [1 -1], 2, {'tansig', 'tansig'});
inNet.trainParam.epochs = 3000;
inNet.trainParam.max_fail = 20;
inNet.trainParam.show = 1000000;
inNet.trainParam.batchSize = 1000;
inNet.trainParam.useSP = true;

%Training the networks to be compared.
tic
[spNet, spEvo] = ntrain(inNet, inTrn, inVal);
toc

inNet.trainParam.useSP = false;
tic
[net, mseEvo] = ntrain(inNet, inTrn, inVal);
toc

%Generating the testing outputs.
out = nsim(net, inTst);
spOut = nsim(spNet, inTst);

%First analysis: RoC.
[sp, cut, det, fa] = genROC(out{1}, out{2});
[maxSP idx] = max(sp);
[spSP, spCut, spDet, spFa] = genROC(spOut{1}, spOut{2}); 
[spMaxSP spIdx] = max(spSP);
 
figure;

subplot(2,2,1);
plot(fa, det, 'r-', spFa, spDet, 'k-');
legend(sprintf('MSE Stop (SP = %f)', maxSP), sprintf('SP Stop (SP = %f)', spMaxSP), 'Location', 'SouthEast');
hold on;
plot(fa(idx), det(idx), 'rx', spFa(spIdx), spDet(spIdx), 'kx');
title('RoC Between the Algorithms');
xlabel('False Alarm (%)');
ylabel('Detection (%)');

subplot(2,2,2);
plot(mseEvo.epoch, mseEvo.mse_trn, 'b-', mseEvo.epoch, mseEvo.mse_val,'r-', spEvo.epoch, spEvo.mse_trn, 'k-', spEvo.epoch, spEvo.sp_val,'m-');
legend('Trn (MSE)', 'Val (MSE)', 'Trn (SP)', 'Val (SP)');
title('Training Evolution');
xlabel('Epoch');
ylabel('MSE');

%Second analysis: output distribution.
nBims = 200;
subplot(2,2,3);
histLog(out, nBims);
title('Output Distribution for the MSE Version')
xlabel('Network Output');
subplot(2,2,4);
histLog(spOut, nBims);
title('Output Distribution for the SP Version')
xlabel('Network Output');
