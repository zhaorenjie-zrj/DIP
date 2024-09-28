import gradio as gr
import cv2
import numpy as np
import math

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
#函数转换2x3仿射矩阵到3x3的矩阵乘法
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
#基于用户输入应用转换的函数
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    ##将图像从PIL格式转换为NumPy数组
    image = np.array(image)
    # Pad the image to avoid boundary issues
    #填充图像以避免边界问题
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform (应用合成变换)
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    scale=1/scale
    scale_m=np.array([[scale,0,0],[0,scale,0]])
    scale_m=to_3x3(scale_m)
    rotation=rotation/180*math.pi
    rotation=rotation
    rotation_m=np.array([[math.cos( rotation),-math.sin( rotation),0],[math.sin( rotation),math.cos( rotation),0]])
    rotation_m=to_3x3(rotation_m)
    translation_x=-translation_x
    translation_y=translation_y
    translation_m=np.array([[1,0,translation_x],[0,1,translation_y]])
    translation_m=to_3x3(translation_m)
    if(flip_horizontal==1):
        flip_horizontal=-1
    else:
        flip_horizontal=1  
    flip_horizontal_m=np.array([[flip_horizontal,0,0],[0,1,0]])
    flip_horizontal_m=to_3x3(flip_horizontal_m)
    composition_transform=scale_m@rotation_m@translation_m@flip_horizontal_m
    for i in range(-image.shape[1]//2,image.shape[1]//2):
        for j in range(-image.shape[0]//2,image.shape[0]//2):
            transform=composition_transform@np.array([[i],[j],[1]])
            k=round(transform[0][0])
            l=round(transform[1][0])
            if(((image.shape[1]//2+k)<image.shape[1]) and ((image.shape[0]//2+l)<(image.shape[0])) and ((image.shape[1]//2+k)>=0) and ((image.shape[0]//2+l)>=0)):
                transformed_image[image.shape[0]//2+j][image.shape[1]//2+i]=image[image.shape[0]//2+l][image.shape[1]//2+k]
                #transformed_image[image.shape[0]//2+l][image.shape[1]//2+k]=image[image.shape[0]//2+j][image.shape[1]//2+i]
    
    return transformed_image

# Gradio Interface(gradio接口)
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        #定义布局
        with gr.Row():
            # Left: Image input and sliders
            #左：图像输入和滑块
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        #自动更新输出时，任何滑块和复选框的改变
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        #链接输入到转换函数
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
#启动gradio界面
interactive_transform().launch(share=True)