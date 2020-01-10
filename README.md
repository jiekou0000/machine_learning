基本概念(sklearn -- tensorflow)：
    https://www.jianshu.com/p/0837b7c6ce10
    https://www.jianshu.com/p/145c09418035
    
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
 
    
参考：
    https://www.jianshu.com/p/0837b7c6ce10  // sklearn和tensorflow区别
    https://www.jianshu.com/p/145c09418035  // 从ML到DL
    https://scikit-learn.org/stable/index.html  // sklearn官网
    https://sklearn.apachecn.org/#/  // sklearn中文文档
    https://blog.csdn.net/u014248127/article/details/78885180  // sklearn库的学习框架
    https://blog.csdn.net/u014410989/article/details/89947128  // python开发包的关系及学习资料
    https://blog.csdn.net/kingzone_2008/article/details/81838271  // python开发包图谱
    https://blog.csdn.net/gdkyxy2013/article/details/80230248  // python开发包的基础使用
    https://blog.csdn.net/m0_37922734/article/details/80287822  // ml相关包学习工具经验
    https://blog.csdn.net/sxeric/article/details/102677417  // ml线性回归算法预测股票走势
    
    
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

    
文档(本工程为0.21.3版本)：
    https://scikit-learn.org/stable/  官方文档(查阅对应版本)
    https://sklearn.apachecn.org  中文文档
    
   
安装anaconda
    清华大学开源软件镜像站：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
    具体安装操作可以参考：https://blog.csdn.net/tz_zs/article/details/73459800

   配置anaconda仓库的镜像
        打开Anaconda Prompt，输入以下命令
            conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
            conda config --set show_channel_urls yes
    
   查看sklearn版本：
        >python
        >>>import sklearn
        >>>sklearn.__version__
       

机器学习
    类型：
        监督学习：
            任务(方式)：
                分类：线性、决策树、SVM、KNN，朴素贝叶斯；集成分类：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees
                回归：线性、决策树、SVM、KNN ；集成回归：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees
        无监督学习：
            任务(方式)：
                聚类：k均值（K-means）、层次聚类（Hierarchical clustering）、DBSCAN
                降维：LinearDiscriminantAnalysis、PCA
        半监督学习：
            ...
        
   主要步骤中sklearn应用：
       数据集、数据预处理、选择模型并训练、模型评分、模型的保存与恢复
           
    
    
    
    
    

境界：
    了解算法的过程和作用
    能把算法运用到实践中 *
    对算法的推导融会贯通


目标：-- (scikit-learn库，模型评估与选择后)掌握一种机器学习算法(了解算法的过程和作用)并把算法运用到实践:平安银行(000001.SZ)股价预测，其它算法的基本了解:了解算法的过程和作用
    数学：
        《线性代数》最基本的矩阵知识，《概率论与统计》，其余在机器学习算法的学习过程中查漏补缺；
    编程：
        Python编程，平均每周投入了5－8小时，最后都会写一个100-200行的Python程序保持coding的感觉。
    机器学习相关的包：
        Numpy、Pandas 和 scikit-learn。总共花了5小时去熟悉Numpy和Pandas，并没有专门花时间学习scikit-learn。
    机器学习－初体验：
        使用scikit-learn的包做一些case-study类似的项目。github上类似的项目很多，比如最经典的房价预测、文件分类、歌曲推荐等。总共投入时间大约15小时。
        体验完scikit-learn的算法包之后，我接着学习了机器学习中一些基本的“模型评估与选择”的方法论，比如confusion matrix、accuracy、precision、recall、fbeta-score、cross-validation、kFold等，并应用到“房价预测”这个项目中。总共投入时间大约15小时。
    机器学习－算法学习：
        机器学习算法全部是“监督学习”的范畴，包括线性回归、逻辑回归、决策树、支持向量机、朴素贝叶斯和组合式学习(AdaBoost)。
        对于上述的六种监督式机器学习算法，目前的程度只达到了理解原理、理解数学推导、调用scikit-learn中的算法包完成了一个简单的教程项目。总共投入时间大约为40小时。
        目前对这六种算法掌握的程度肯定是不够的，能够自行推导数学过程 和 用python、numpy、pandas实现粗略版的算法 是3月底这几天的目标。（未完成）
    能够掌握下图中所示的内容，可算作基本掌握了一个机器学习算法：
        z_goal.jpg  -- 原理及实际运用
    总结：
        深入学习上述六种监督式机器学习算法；
        除了监督式机器学习，之后还有非监督式机器学习、强化学习；
        除了机器学习，之后还有深度学习、无人车、图像识别等各种相关深入技术；
        
  熟悉scikit-learn库中算法及API:过程和作用及使用场景和使用方法；
  以预测平安银行(000001.SZ)股价走势为demo将机器学习算法运用到实践
  
  通过机器学习的线性回归算法预测股票走势（Python实现）
        