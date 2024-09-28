import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    points_src_m=np.array(points_src)
    points_dst_m=np.array(points_dst)
    for k in range(image.shape[1]):
        for l in range(image.shape[0]):
            v=np.array([k,l])
            w=np.zeros(points_src_m.shape[0])
            total_w=0
            flag=0
            total_wp=np.zeros((1,2))
            total_wq=np.zeros((1,2))
            #求权重算子w与p_*、q_*
            for i in range(points_src_m.shape[0]):
                if(i<points_dst_m.shape[0]):
                    w[i]=(points_src_m[i][0]-k)**2+(points_src_m[i][1]-l)**2
                    if(w[i]==0):
                        flag=1
                        break
                    else:
                        w[i]=1/w[i]
                        total_w=total_w+w[i]
                        total_wp=total_wp+w[i]*points_src_m[i]
                        total_wq=total_wq+w[i]*points_dst_m[i]
            if(flag==1):
                flag==0
                continue
            p_asterisk=total_wp/total_w
            q_asterisk=total_wq/total_w
            p=np.zeros((1,2))
            q=np.zeros((1,2))
            #求2*2矩阵∑p_t*w*p与∑w*p*q
            total_pwp=np.zeros((2,2))
            total_wpq=np.zeros((2,2))
            for i in range(points_src_m.shape[0]):
                if(i<points_dst_m.shape[0]):
                    p=points_src_m[i]-p_asterisk
                    p=np.array(p)
                    q=points_dst_m[i]-q_asterisk
                    q=np.array(q)
                    total_pwp=total_pwp+w[i]*(np.transpose(p)@p)
                    total_wpq=total_wpq+w[i]*(np.transpose(p)@q)
            fa_v=(v-q_asterisk)@np.linalg.pinv(total_wpq)@total_pwp+p_asterisk
            #fa_v=(v-p_asterisk)@np.linalg.pinv(total_pwp)@total_wpq+q_asterisk
            fa_v=np.round(fa_v)
            fa_v=fa_v.astype(np.int)
           # print('%d   %d  %d  %d',fa_v[0],fa_v[1],k,l)
            if((-1<fa_v[0][0]<(image.shape[1])) and (-1<fa_v[0][1]<(image.shape[0])) ):
                #warped_image[fa_v[0][1]][fa_v[0][0]]=image[l][k]
                warped_image[l][k]=image[fa_v[0][1]][fa_v[0][0]]
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
