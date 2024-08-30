from PIL import Image, ImageDraw


def crop_circle(image_path, radius):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size

    # 创建一个空白的图像，用于绘制圆形遮罩
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((width / 2 - radius, height / 2 - radius, width / 2 + radius, height / 2 + radius), fill=255)

    # 创建一个与原始图像相同大小的透明图像
    result = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # 将遮罩应用到原始图像上，裁剪出圆形，并将非圆形部分设置为透明
    result.paste(img, (0, 0), mask)

    # 裁剪出圆形部分
    result = result.crop((width / 2 - radius, height / 2 - radius, width / 2 + radius, height / 2 + radius))

    # 保存裁剪后的圆形图像
    result.save("yjtp-modified.png")


# 指定图片路径和半径
image_path = "./icon/tigerking.png"
radius = 469

# 调用函数进行裁剪
crop_circle(image_path, radius)
