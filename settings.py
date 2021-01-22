def tags(names):
    new_list = []
    for name in names:
        new_list.append(tags_map[name])
    return new_list


tags_list = [
              # "mouse", "鼠标",
              # "person", "人",
              "cell phone", "手机",
            ]

tags_map = {
  "person": "人",
  "bicycle": "自行车",
  "car": "汽车",
  "motorcycle": "摩托车",
  "airplane": "飞机",
  "bus": "公共汽车",
  "train": "火车",
  "truck": "卡车",
  "boat": "船",
  "traffic light": "交通灯",
  "fire hydrant": "消防栓",
  "stop sign": "停车标志",
  "parking meter": "停车计时器",
  "bench": "长凳",
  "bird": "鸟",
  "cat": "猫",
  "dog": "狗",
  "horse": "马",
  "sheep": "绵羊",
  "cow": "奶牛",
  "elephant": "大象",
  "bear": "熊",
  "zebra": "斑马",
  "giraffe": "长颈鹿",
  "backpack": "背包",
  "umbrella": "雨伞",
  "handbag": "手提包",
  "tie": "领带",
  "suitcase": "手提箱",
  "frisbee": "飞盘",
  "skis": "双滑雪板",
  "snowboard": "单滑雪板",
  "sports ball": "运动球",
  "kite": "风筝",
  "baseball bat": "棒球棒",
  "baseball glove": "棒球手套",
  "skateboard": "滑板",
  "surfboard": "冲浪板",
  "tennis racket": "网球拍",
  "bottle": "瓶子",
  "wine glass": "酒杯",
  "cup": "杯子",
  "fork": "叉子",
  "knife": "刀",
  "spoon": "勺子",
  "bowl": "碗",
  "banana": "香蕉",
  "apple": "苹果",
  "sandwich": "三明治",
  "orange": "橘子",
  "broccoli": "花椰菜",
  "carrot": "胡萝卜",
  "hot dog": "热狗",
  "pizza": "披萨",
  "donut": "甜甜圈",
  "cake": "蛋糕",
  "chair": "椅子",
  "couch": "沙发",
  "potted plant": "盆栽植物",
  "bed": "床",
  "dining table": "餐桌",
  "toilet": "厕所",
  "tv": "电视",
  "laptop": "笔记本电脑",
  "mouse": "鼠标",
  "remote": "遥控器",
  "keyboard": "键盘",
  "cell phone": "手机",
  "microwave": "微波炉",
  "oven": "烤箱",
  "toaster": "烤面包机",
  "sink": "水槽",
  "refrigerator": "冰箱",
  "book": "书",
  "clock": "时钟",
  "vase": "花瓶",
  "scissors": "剪刀",
  "teddy bear": "泰迪熊",
  "hair drier": "吹风机",
  "toothbrush": "牙刷"
}