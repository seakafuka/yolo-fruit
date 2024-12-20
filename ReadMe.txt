# 项目介绍
    本项目基于yolov8n模型实现，前端使用gradio框架



# 目录结构描述
    ├── frontend            // 包含前端运行文件代码
    
    │   ├── demo.ipynb      // 前端运行后，点击网页链接进入前端测试
    
    ├── label_data           // 本小组图片标注

    │   ├── Annotations         // 包含标注的xml文件

    │   ├── images          // 图片
    
    │   ├── imageSets

    │   ├── labelmap.txt       
    
    ├── test             // 老师所给测试集
    
    ├──Txtojson                // 包含项目格式转化文件

    │   ├── class.txt    

    └── best.pt               //包含训练好的模型参数

    └── data.yaml           //模型训练参数设置，"""需要更改"""
    
    └── predict.py           //预测文件

    └── Readme.md           //介绍文档


# 使用说明

## 一、安装环境依赖

    !pip install ultralytics
    !pip install gradio

## 二、修改data.yaml文件

    将"val: E:\vscode_code\apple_and banana_final\test\images"中的
    "E:\vscode_code\apple_and banana_final"改为你电脑上的文件地址

## 三、 运行predict.py文件和frontend文件夹中的demo.ipynb

    注：iou指标计算基于第一次valid结果，因此如果第一遍没有跑通，后续再跑需要删除文件夹目录中新产生的runs文件夹，这样才可以得到正确的iou指标。其他指标不受影响

