Reversion History     
========================================      
* **2019.11.16**     
1.由于tensorflow中某些函数不能被openvino转换，因此额外添加适应于
openvino的转换的feature map合并函数；       
2.测试openvino对转换后模型的支持--通过；       
3.利用openvino对`.pb`文件进行转换，并编写openvino下的检测代码；            
* **2019.11.11**     
1.添加将检测结果矩形框尺寸映射到原始图像上函数         
* **2019.11.8-2**      
1.将模型转换为`.pb`文件，修改主干网络后，原转换方式存在bug，故进行修改;同时，网络输出节点为拼接后的最终检测结果，而非3个feituremap的结果;      
2.对转换后的`.pb`文件进行测试，初步验证，可以进行正常预测工作     
* **2019.11.8**     
1.nms过滤后，选择检测结果之前进行排序操作       
2.修复`tf1.12.0`中`tf.image.non_max_suppression`导致的程序崩溃bug    
* **2019.11.5**      
1.修改backbone为mobilenetV2       
* **2019.10.29**        
1.将原网络输入的3个feature map的预测框合并为一个输出;    
2.在`tf.summary()`中添加预测结果框的显示            
* **2019.10.27**     
增加对3个pred_box的处理，通过置信度处理，nms过滤，使其返回为一个box，待验证    
* **2019.10.25**     
1.添加TFRecord数据生成与读取；        
2.summary添加image及ground truth box的显示；       
* **2019.8.22**      
1.`_train.py`添加接着上次训练结果训练功能，添加`tf.summary.image`功能；      
2.添加阅读注释:边框预测函数`def decode(self, conv_output, anchors, stride)`     
* **2019.8.21**   
1.添加注释；    
2.更新`README.md`       
* **2019.8.19**      
1.添加阅读注释；    
* **2019.8.15-2**             
1.add image to summary;          
* **2019.8.15-1**    
1.添加raccoon数据集进行训练测试；           
* **2019.8.15**   
1.完成代码;    
2.进行视频测试;      
* **2019.6.5**    
1.完成代码文件`core/yolov3.py`,`core/dataset.py`,至此，文件夹`core`创建完毕;     
* **2019.5.22**   
1.创建文件夹`data/classes`，其中存放类别标签名称；    
2.完成代码文件`core/config.py`;      
* **2019.5.20**     
1.创建仓库;   
2.创建文件夹`core`存放网络核心代码，完成代码文件`core/common.py`,`core/backbone.py`代码的创建;    
