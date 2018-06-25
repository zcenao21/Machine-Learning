clear all
close all
clc

alpha=0.2; %梯度下降学习率
ite_time=100;

% 首先作出原始数据分布图
data=load('ex1data1.txt');
x=data(:,1);
y=data(:,2);

figure
plot(x,y,'blue+')

% 对数据归一化
x_ave=sum(x)/length(x);
x_range=max(x)-min(x);
x_processed=(x-x_ave)/x_range;

y_ave=sum(y)/length(y);
y_range=max(y)-min(y);
y_processed=(y-y_ave)/y_range;

% figure
% plot(x_processed,y_processed,'red+')

% 梯度下降训练
theta=zeros(2,1); %设置初始值
theta(1)=1;
theta(2)=1;
for j=1:ite_time
    theta_1_temp=theta(1);
    theta(1)=theta(1)-alpha*sum((theta(1)+theta(2)*x_processed-y_processed))/length(x);
    theta(2)=theta(2)-alpha*sum((theta_1_temp+theta(2)*x_processed-y_processed).*x_processed)/length(x);

    % 保存每次训练完成后的数据，用于三维作图
    theta_1_save(j)=theta(1);
    theta_2_save(j)=theta(2);
    temp1=0;
    for k=1:length(x)
        temp1=temp1+(theta(1)+theta(2)*x_processed(k)-y_processed(k)).^2;
    end
    JTheta_save(j)=1/(2*length(x))*temp1;

    % 计算每次训练完成后的误差，error_processed为平均误差
    error(j,:)=theta(1)+theta(2)*x_processed-y_processed;
    error_processed(j)=sum(abs(error(j,:)))/length(x);
end

% 作出归一化的数据拟合图
x_plot=min(x_processed):0.05:max(x_processed);
figure
plot(x_processed,y_processed,'red+')
hold on
plot(x_plot,theta(2)*x_plot+theta(1),'green-','linewidth',2) 
hold on

% 平均误差随运行批次的变化情况
figure
plot(error_processed,'r-*')

% 作出原数据拟合结果
x_plot_real=min(x):0.05:max(x);
figure
plot(x,y,'blue+')
hold on
plot(x_plot_real,y_range/x_range*theta(2)*(x_plot_real-x_ave)+theta(1)*y_range+y_ave,'r-')

% 针对不同theta的代价函数
for theta1=-1:0.05:1
    i=int16((theta1+1)*20+1);
    for theta2=0:0.05:2
        j=int16((theta2+0)*20+1);
        JTheta(i,j)=1/(2*length(x))*sum((theta1+theta2*x_processed-y_processed).^2);
    end
end

% 作出代价函数三维图形
scrsz = get(0,'ScreenSize');  % 是为了获得屏幕大小，Screensize是一个4元素向量[left,bottom, width, height]
figure
set(gcf,'Position',scrsz);
surf(-1:0.05:1,0:0.05:2,JTheta')
hold on

% 设置三维图角度
view(-25.5,40);
% 在三维图上描点，显示梯度下降法的代价函数变化情况
for i=1:2:ite_time
    j=int16((i+1)/2); %间隔为2取图形作gif图形
    plot3(theta_1_save(i),theta_2_save(i),JTheta_save(i),'r.-','Markersize',15)
    hold on
%     picname=[num2str(j) '.fig'];%保存的文件名
%     saveas(gcf,picname)
end
% 
% % 生成gif
% stepall=40;
% for i=1:stepall
%     picname=[num2str(i) '.fig'];
%     open(picname);
%     set(gcf,'outerposition',get(0,'screensize'));% matlab窗口最大化
%     frame=getframe(gcf);  
%     im=frame2im(frame);%制作gif文件，图像必须是index索引图像  
%     [I,map]=rgb2ind(im,20);          
%     if i==1
%         imwrite(I,map,'LinearRegression.gif','gif', 'Loopcount',inf,'DelayTime',0.1);%第一次必须创建！
%     elseif i==stepall
%         imwrite(I,map,'LinearRegression.gif','gif','WriteMode','append','DelayTime',0.1);
%     else
%         imwrite(I,map,'LinearRegression.gif','gif','WriteMode','append','DelayTime',0.1);
%     end;  
%     close all;
% end