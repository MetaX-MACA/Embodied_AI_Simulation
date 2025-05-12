import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
from shutil import copyfile, copytree

scene_template_path = 'data/scene_template.xml'

root_path = os.path.dirname(os.path.abspath(scene_template_path))

assets_path = os.path.join(root_path, 'assets')
meshes_path = os.path.join(root_path, 'meshes')
textures_path = os.path.join(root_path, 'textures')
franka_pada_path = os.path.join(root_path, 'third_party')
scene_bash_path = os.path.join(root_path, 'scene_base.xml')

def create_link(target, link_name):
    try:
        os.symlink(target, link_name)
        # print(f"软链接已创建：{link_name} -> {target}")
    except FileExistsError:
        print(f"软链接已存在：{link_name}")
    except PermissionError:
        print(f"权限不足，无法创建软链接。{link_name}")
    except Exception as e:
        print(f"创建软链接时出错：{e}")


def remove_empty_line(pretty_xml):
    # 保留最多一个空行
    lines = pretty_xml.splitlines()
    cleaned_xml = []
    for i, line in enumerate(lines):
        if line.strip() == "" and (i == 0 or lines[i - 1].strip() == ""):
            continue
        cleaned_xml.append(line)
    cleaned_xml = "\n".join(cleaned_xml)
    return cleaned_xml


def fill_mujoco_template(obj_dict):
    # 读取模板文件
    with open(scene_template_path, "r", encoding="utf-8") as template_file:
        template_content = template_file.read()

    scene_xml = ET.fromstring(template_content)

    # 添加每个对象的 asset include
    for obj_name, obj_data in obj_dict.items():
        ET.SubElement(scene_xml, "include", file=obj_data['asset_path'])

    # 查找 worldbody 元素
    worldbody = scene_xml.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(scene_xml, "worldbody")

    # 添加每个对象的 body 元素
    for obj_name, obj_data in obj_dict.items():
        body = ET.SubElement(worldbody, "body", name=obj_name, pos=obj_data['pos'], euler="0 0 -1.57")
        # 添加 chain include 元素
        ET.SubElement(body, "include", file=obj_data['chain_path'])
        # 如果需要 freejoint
        if obj_data['movable']:
            ET.SubElement(body, "freejoint")

    # 转换为字符串
    xml_string = ET.tostring(scene_xml, encoding='utf-8', method='xml')
    # 美化格式
    reparsed  = xml.dom.minidom.parseString(xml_string)
    pretty_xml = reparsed.toprettyxml(indent="    ") # 每一级子元素的缩进为 4 个空格
    pretty_xml = remove_empty_line(pretty_xml)

    return pretty_xml


def generate_mujoco_scene_xml(config, task_config_path, task_name):
    """
    根据场景配置创建mujoco场景xml文件
    """
    # 创建task目录，并将资产软连接到task目录中
    task_dir = os.path.join(os.path.dirname(task_config_path), f'task_{task_name.replace(" ", "_")}')
    task_scene_dir = os.path.join(task_dir, "scene")
    os.makedirs(task_scene_dir, exist_ok=True)
    
    # 使用绝对路径创建软链接，不利于任务迁移
    # create_link(assets_path, os.path.join(task_scene_dir, 'assets'))
    # create_link(meshes_path, os.path.join(task_scene_dir, 'meshes'))
    # create_link(textures_path, os.path.join(task_scene_dir, 'textures'))
    # create_link(franka_pada_path, os.path.join(task_scene_dir, 'third_party'))
    # create_link(scene_bash_path, os.path.join(task_scene_dir, 'scene_base.xml'))

    # 转为相对于task_scene_dir的相对路径(mujoco加载会报错)
    # rel_assets_path = os.path.relpath(assets_path, task_scene_dir)
    # rel_meshes_path = os.path.relpath(meshes_path, task_scene_dir)
    # rel_textures_path = os.path.relpath(textures_path, task_scene_dir)
    # rel_franka_pada_path = os.path.relpath(franka_pada_path, task_scene_dir)
    # rel_scene_bash_path = os.path.relpath(scene_bash_path, task_scene_dir)
    # create_link(rel_assets_path, os.path.join(task_scene_dir, 'assets'))
    # create_link(rel_meshes_path, os.path.join(task_scene_dir, 'meshes'))
    # create_link(rel_textures_path, os.path.join(task_scene_dir, 'textures'))
    # create_link(rel_franka_pada_path, os.path.join(task_scene_dir, 'third_party'))
    # create_link(rel_scene_bash_path, os.path.join(task_scene_dir, 'scene_base.xml'))

    # 为避免mujoco中引用路径错误，改为直接复制目录和文件
    copytree(assets_path, os.path.join(task_scene_dir, 'assets'))
    copytree(meshes_path, os.path.join(task_scene_dir, 'meshes'))
    copytree(textures_path, os.path.join(task_scene_dir, 'textures'))
    copytree(franka_pada_path, os.path.join(task_scene_dir, 'third_party'))
    copyfile(scene_bash_path, os.path.join(task_scene_dir, 'scene_base.xml'))

    # 遍历得到物体信息
    objs = {}
    ref_assets_path = './assets'
    for obj in config:
        if 'name' in obj.keys():
            obj_name = obj['name'].lower()
            center = obj['center']
            pos = center.replace("(", "").replace(")", "").replace(",", " ")
            objs[obj_name] = {'pos': pos, 
                              'asset_path': os.path.join(ref_assets_path, f"{obj_name}_asset.xml"), 
                              'chain_path': os.path.join(ref_assets_path, f"{obj_name}_chain.xml"), 
                              'movable': obj['movable'], 
                              }
    # 向base场景中填充新增物体
    pretty_xml = fill_mujoco_template(objs)

    # 保存xml文件
    scene_xml_path = os.path.join(task_scene_dir, 'scene.xml')
    with open(scene_xml_path, "w", encoding="utf-8") as output_file:
        output_file.write(pretty_xml)

    return scene_xml_path