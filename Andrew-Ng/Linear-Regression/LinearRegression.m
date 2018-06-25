clear all
close all
clc

alpha=0.2; %�ݶ��½�ѧϰ��
ite_time=100;

% ��������ԭʼ���ݷֲ�ͼ
data=load('ex1data1.txt');
x=data(:,1);
y=data(:,2);

figure
plot(x,y,'blue+')

% �����ݹ�һ��
x_ave=sum(x)/length(x);
x_range=max(x)-min(x);
x_processed=(x-x_ave)/x_range;

y_ave=sum(y)/length(y);
y_range=max(y)-min(y);
y_processed=(y-y_ave)/y_range;

% figure
% plot(x_processed,y_processed,'red+')

% �ݶ��½�ѵ��
theta=zeros(2,1); %���ó�ʼֵ
theta(1)=1;
theta(2)=1;
for j=1:ite_time
    theta_1_temp=theta(1);
    theta(1)=theta(1)-alpha*sum((theta(1)+theta(2)*x_processed-y_processed))/length(x);
    theta(2)=theta(2)-alpha*sum((theta_1_temp+theta(2)*x_processed-y_processed).*x_processed)/length(x);

    % ����ÿ��ѵ����ɺ�����ݣ�������ά��ͼ
    theta_1_save(j)=theta(1);
    theta_2_save(j)=theta(2);
    temp1=0;
    for k=1:length(x)
        temp1=temp1+(theta(1)+theta(2)*x_processed(k)-y_processed(k)).^2;
    end
    JTheta_save(j)=1/(2*length(x))*temp1;

    % ����ÿ��ѵ����ɺ����error_processedΪƽ�����
    error(j,:)=theta(1)+theta(2)*x_processed-y_processed;
    error_processed(j)=sum(abs(error(j,:)))/length(x);
end

% ������һ�����������ͼ
x_plot=min(x_processed):0.05:max(x_processed);
figure
plot(x_processed,y_processed,'red+')
hold on
plot(x_plot,theta(2)*x_plot+theta(1),'green-','linewidth',2) 
hold on

% ƽ��������������εı仯���
figure
plot(error_processed,'r-*')

% ����ԭ������Ͻ��
x_plot_real=min(x):0.05:max(x);
figure
plot(x,y,'blue+')
hold on
plot(x_plot_real,y_range/x_range*theta(2)*(x_plot_real-x_ave)+theta(1)*y_range+y_ave,'r-')

% ��Բ�ͬtheta�Ĵ��ۺ���
for theta1=-1:0.05:1
    i=int16((theta1+1)*20+1);
    for theta2=0:0.05:2
        j=int16((theta2+0)*20+1);
        JTheta(i,j)=1/(2*length(x))*sum((theta1+theta2*x_processed-y_processed).^2);
    end
end

% �������ۺ�����άͼ��
scrsz = get(0,'ScreenSize');  % ��Ϊ�˻����Ļ��С��Screensize��һ��4Ԫ������[left,bottom, width, height]
figure
set(gcf,'Position',scrsz);
surf(-1:0.05:1,0:0.05:2,JTheta')
hold on

% ������άͼ�Ƕ�
view(-25.5,40);
% ����άͼ����㣬��ʾ�ݶ��½����Ĵ��ۺ����仯���
for i=1:2:ite_time
    j=int16((i+1)/2); %���Ϊ2ȡͼ����gifͼ��
    plot3(theta_1_save(i),theta_2_save(i),JTheta_save(i),'r.-','Markersize',15)
    hold on
%     picname=[num2str(j) '.fig'];%������ļ���
%     saveas(gcf,picname)
end
% 
% % ����gif
% stepall=40;
% for i=1:stepall
%     picname=[num2str(i) '.fig'];
%     open(picname);
%     set(gcf,'outerposition',get(0,'screensize'));% matlab�������
%     frame=getframe(gcf);  
%     im=frame2im(frame);%����gif�ļ���ͼ�������index����ͼ��  
%     [I,map]=rgb2ind(im,20);          
%     if i==1
%         imwrite(I,map,'LinearRegression.gif','gif', 'Loopcount',inf,'DelayTime',0.1);%��һ�α��봴����
%     elseif i==stepall
%         imwrite(I,map,'LinearRegression.gif','gif','WriteMode','append','DelayTime',0.1);
%     else
%         imwrite(I,map,'LinearRegression.gif','gif','WriteMode','append','DelayTime',0.1);
%     end;  
%     close all;
% end