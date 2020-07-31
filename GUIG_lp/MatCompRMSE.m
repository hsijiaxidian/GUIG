function [ RMSE ] = MatCompRMSE( U, V, row, col, gndtruth )

predict = partXY(U', V', row, col, length(gndtruth))';

predict = predict - gndtruth;
predict = sum(predict.^2);

RMSE = sqrt(predict/length(gndtruth));

end

