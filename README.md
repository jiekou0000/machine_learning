基本概念(sklearn -- tensorflow)：
    https://www.jianshu.com/p/0837b7c6ce10
    https://www.jianshu.com/p/145c09418035
    
    
资源：
    （1）基础篇
        书籍：
            1.统计机器学习。李航
            2.机器学习。周志华
        视频：
            1.机器学习。斯坦福。吴恩达
            2.Tom Mitchell(CMU)机器学习
    （2）升级篇
        书籍:
            1.机器学习实战
            2.深度学习-AI圣经
        视频：
            1.Learning from Data
            2.機器學習基石
    （3）实战篇

机器学习的方法包括：
    监督学习 supervised learning;
    非监督学习 unsupervised learning;
    半监督学习 semi-supervised learning;
    强化学习 reinforcement learning;
    遗传算法 genetic algorithm.

特征工程的一般步骤如下：
    确定任务：根据实际业务确定需要解决的问题
    数据选择：整合数据
    预处理：涉及数据清洗等一系列操作
    构造特征：将数据转换为有效的特征，常用归一化、标准化、特征离散处理等方法
    模型使用：计算模型在该特征上的准确率
    上线效果：通过在线测试来判断特征是否有效
    进行迭代
  除去语音和图像等特定场景，对于大部分生活中的机器学习项目，由于没有足够的训练数据支撑，我们还无法完全信任算法自动生成的特征，因而基于人工经验的特征工程依然是目前的主流。这就需要建模人员对实际业务有较深的理解和把控
  深度学习便是解决特征提取问题的一个机器学习分支。
    
    
文档：
    https://scikit-learn.org/stable/  官方文档
    https://sklearn.apachecn.org  中文文档
    
   
安装anaconda并新建项目
       

