import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
seaborn.set(font=myfont.get_name())

#该数据是很多房子的特征，对应的房价
data=pd.read_csv('data_train.csv',delimiter=',',encoding='GBK')
#查找下哪些是缺失值,然后也看下数量;同时也看下哪些不是缺失值
data_=data.isnull().sum().sort_values(ascending=False)
data_notnull=data_[data_==0]#缺失值一个都没有
data_isnull=data_[data_!=0]#缺失值有，所以不为0
data_notnull.index
data_isnull.index

# =============================================================================
# 由于数据太多，简单点，我就选择指标：
#1.街道连接距离（因为有缺失值）数值型变量，作为自变量
#2.房价作为因变量
#回归消除异常值
# =============================================================================
x=data.街道连接距离
y=data.房价

# =============================================================================
#.第一种方法：肉眼观察大致删除异常值 
# =============================================================================
plt.figure(figsize=(12,6),dpi=600)
seaborn.scatterplot(x,y)
plt.annotate('异常值', fontsize=20,xy=[310,200000], xytext=[270,350000], arrowprops={'facecolor':'red'})
plt.ylabel('房价')
plt.xlabel('街道连接距离')
plt.title('寻找异常值')
plt.show()
#删除异常值
x_del_abnormal=x.drop(x[x>300].index)


# =============================================================================
# 第二种方法：分位数确定异常值，此处用箱线图
# =============================================================================
#绘制初始图

#1.如果画单个变量的
IQR=1.5*(price.quantile(0.75)-price.quantile(0.25))
top=price.quantile(0.75)+IQR
bottom=price.quantile(0.25)-IQR
price_new=price[(price<top)&(price>bottom)]
#绘制对比
fig,ax=plt.subplots(2,1,dpi=600)
seaborn.boxplot(x,ax=ax[0])
ax[0].set_xlabel('街道连接距离(原始数据之前)')
seaborn.boxplot(x_new,ax=ax[1])
ax[1].set_xlabel('街道连接距离(删除异常值之后)')
fig.tight_layout(pad=0.2)



#2.如果是画两个变量间关系的
data_concatence=pd.concat([x,y],axis=1)
plt.figure(figsize=(12,6),dpi=600)
seaborn.boxplot(data=data_concatence,x='街道连接距离',y='房价')
plt.xlabel('街道连接距离')
plt.ylabel('房价')
plt.title('删除异常值之前的箱线图')
plt.xticks(rotation=90)
plt.show()

#data就是data_concatence,x是街道连接距离，y是房价
#分类每个x轴上的值       
data_groupby=data_concatence.groupby(data_concatence['街道连接距离'])
#过滤
def filterfuc_max(x):
    return x-(x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25)))
a=data_groupby['房价'].transform(filterfuc_max)#计算分类后各原始点与最高点的差值
m=a[a>0].index #获得超过最高点的异常值索引

def filterfuc_min(x):
    return (x.quantile(0.25)-1.5*(x.quantile(0.75)-x.quantile(0.25)))-x
b=data_groupby['房价'].transform(filterfuc_min)#同理获得分类后各原始点与最低值的差值
n=b[b>0].index#获得低于最低点的异常值索引
delete_index=m.append(n) #将异常值索引值合并
data_concatence_drop = data_concatence.drop(index=delete_index) #删除异常值

#绘制删除异常值之后的图
plt.figure(figsize=(12,6),dpi=600)
seaborn.boxplot(data=data_concatence_drop,x='街道连接距离',y='房价')
plt.xlabel('街道连接距离')
plt.ylabel('房价')
plt.title('删除异常值的箱线图')
plt.xticks(rotation=90)
plt.show()










