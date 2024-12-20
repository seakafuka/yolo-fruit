from ultralytics import YOLO
import xml.etree.ElementTree as ET
import os
import shutil
import json
from PIL import Image
import numpy as np

def Txtococo(current_dir):
    txt_labels_path=os.path.join(current_dir,'test/labels')
    datasets_img_path=os.path.join(current_dir,'test/images')
    save_path=os.path.join(current_dir,'Txtojson')
    classes_txt=os.path.join(current_dir,'Txtojson/class.txt')
    
    
    with open(classes_txt,'r') as fr:
        lines1=fr.readlines()
    
    categories=[]
    for j,label in enumerate(lines1):
        label=label.strip()
        categories.append({'id':j,'name':label,'supercategory':'None'})
    
    write_json_context=dict()
    write_json_context['info']= {'description': 'For object detection', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021'}
    write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
    write_json_context['categories']=categories
    write_json_context['images']=[]
    write_json_context['annotations']=[]
    
    imageFileList=os.listdir(datasets_img_path)
    
    
    for i,imageFile in enumerate(imageFileList):
        imagePath = os.path.join(datasets_img_path,imageFile)
        image = Image.open(imagePath)
        W, H = image.size
    
        img_context={}
        img_context['file_name']=imageFile
        img_context['height']=H
        img_context['width']=W
        img_context['id'] = imageFile[:-4]
        int_id = img_context['id']
        img_context['license']=1
        img_context['color_url']=''
        img_context['flickr_url']=''
        write_json_context['images'].append(img_context)
    
        txtFile=imageFile[:-4]+'.txt'
    
        with open(os.path.join(txt_labels_path,txtFile),'r') as fr:
            lines=fr.readlines()
        for j,line in enumerate(lines):
            bbox_dict = {}
    
            class_id,x,y,w,h=line.strip().split(' ')
            class_id,x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)
            xmin=(x-w/2)*W
            ymin=(y-h/2)*H
            xmax=(x+w/2)*W
            ymax=(y+h/2)*H
            w=w*W
            h=h*H
            bbox_dict['id']=i*10000+j
            bbox_dict['image_id']=imageFile[:-4]
            bbox_dict['category_id']=class_id
            bbox_dict['iscrowd']=0
            height,width=abs(ymax-ymin),abs(xmax-xmin)
            bbox_dict['area']=height*width
            bbox_dict['bbox']=[xmin,ymin,w,h]
            bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
            write_json_context['annotations'].append(bbox_dict)
    
    name = os.path.join(save_path,"instances_val2017"+ '.json')
    with open(name,'w') as fw:
        json.dump(write_json_context,fw,indent=2)
    
def xmltotxt(xml_dir_path,output_path):
    # XML文件路径
    xml_dir = xml_dir_path
    output_dir = output_path
    try:
        os.mkdir('test\labels')
        os.mkdir('test\images')
    except FileExistsError:
        pass

    # 遍历XML文件
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            
            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图片尺寸
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            
            # 创建YOLOv8格式的标注文件
            yolo_file_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
            with open(yolo_file_path, 'w') as yolo_file:
                # 遍历每个目标对象
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    if class_name not in class_map:
                        continue
                    
                    class_id = class_map[class_name]
                    xmin = float(obj.find('bndbox').find('xmin').text)
                    ymin = float(obj.find('bndbox').find('ymin').text)
                    xmax = float(obj.find('bndbox').find('xmax').text)
                    ymax = float(obj.find('bndbox').find('ymax').text)
                    
                    # 检查宽度和高度是否为零，避免除零错误
                    if width == 0 or height == 0:
                        continue
                    
                    # 计算YOLOv8格式中心点坐标和相对于图片尺寸的宽度和高度
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    # 写入YOLOv8格式标注信息
                    yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def delete_files_with_suffix(folder_path, suffix):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 判断文件是否为指定后缀的文件
        if file_name.endswith(suffix):
            # 删除文件
            os.remove(file_path)

def move(source_folder, destination_folder,file_extension):
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否具有指定的后缀
        if filename.endswith(file_extension):
            # 构建文件的完整路径
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            
            # 移动文件
            shutil.move(source_path, destination_path)

# 定义IoU函数
def calculate_iou(box1, box2):
     # 计算y轴重叠
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    # 计算x轴重叠
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    # 计算xy重叠面积
    inner = 0 if in_w < 0 or in_h < 0 else in_w * in_h
    # 计算两个矩形并及
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inner
    # 计算iou
    iou = inner / union
    return iou

# 加载JSON文件
def load_json_files(file1, file2):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    return data1, data2

def distance(box1,box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    distance = abs(x1_min-x2_min)+abs(y1_min-y2_min)+abs(x1_max-x2_max)+abs(y1_max-y2_max)
    return distance
# 计算两个JSON文件中所有bbox的平均iou
def calculate_average_iou(file1, file2):
    data1, data2 = load_json_files(file1, file2)
    
    # 提取边界框
    ious = []
    apples_ious = []
    banana_ious = []
    orange_ious = []
    for annotation in data1['annotations']:
        image_id = annotation['image_id']
        fruitclass = image_id[:-3]
        box1 = annotation['bbox']
        box2s = []
        for dict in data2:
            if dict['image_id'] == image_id:
                box2s.append(dict['bbox'])
        box2 = []
        min = 1000
        for box in box2s:
            dis = distance(box1,box)
            if dis <= min:
                min= dis
                box2 = box
        iou = calculate_iou(box1, box2)
        ious.append(iou)
        if fruitclass == 'apple':
            apples_ious.append(iou)
        elif fruitclass == 'banana':
            banana_ious.append(iou)
        elif fruitclass == 'orange':
            orange_ious.append(iou)
    
    # 计算平均IoU
    average_iou = np.mean(ious)
    apples_average_ious = np.mean(apples_ious)
    banana_average_ious = np.mean(banana_ious)
    orange_average_ious = np.mean(orange_ious)
    return average_iou,apples_average_ious,banana_average_ious,orange_average_ious


if __name__ == '__main__':
    
    model = YOLO('best.pt') 
    
    #获取当前目录
    current_dir = os.getcwd()
    # 定义类别名称映射字典
    class_map = {
        'apple': 0,
        'banana': 1,
        'orange': 3
        # 添加其他类别映射
    }
    # 在验证集上评估模型
    xmltotxt('test','test')
    delete_files_with_suffix('test','.xml')
    move('test','test/images','jpg')
    move('test','test/labels','txt')
    # xmltotxt('train','train')
    # delete_files_with_suffix('train','.xml')
    # move('train','train/images','jpg')
    # move('train','train/labels','txt')
    metrics = model.val(save_json = True)  # 调整置信度和IoU阈值
    Txtococo(current_dir)
    file_path = os.path.join(current_dir, 'Txtojson/instances_val2017.json')
    predict_path = os.path.join(current_dir,'runs/detect/val/predictions.json')
    average_iou,apple,banana,orange = calculate_average_iou(file_path,predict_path)
    print('                              IOU')
    print(f"Average IoU: {average_iou:>21.2f}")
    print(f"Apple Average IoU: {apple:>15.2f}")
    print(f"Banana Average IoU: {banana:>14.2f}")
    print(f"Orange Average IoU: {orange:>14.2f}")