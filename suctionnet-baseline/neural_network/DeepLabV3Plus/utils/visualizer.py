from visdom import Visdom
import json 

class Visualizer(object):
    """ 可视化工具类, 基于Visdom实现训练过程中的数据可视化
    """
    def __init__(self, port='13579', env='main', id=None):
        # 当前窗口的映射关系, key为窗口标题, value为窗口id
        self.cur_win = {}
        # 初始化Visdom对象
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env
        # 恢复已有窗口, 避免重复创建
        ori_win = self.vis.get_window_data()
        ori_win = json.loads(ori_win)
        # 将已有窗口的标题与id建立映射
        self.cur_win = { v['title']: k for k, v in ori_win.items()  }

    def vis_scalar(self, name, x, y, opts=None):
        """
        可视化标量数据(如loss、accuracy曲线)

        参数说明:
            name (str): 曲线名称
            x (int或list): 横坐标(如迭代次数)
            y (float或list): 纵坐标(如loss值)
            opts (dict): 其他visdom参数
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if self.id is not None:
            name = "[%s]"%self.id + name
        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)

        win = self.cur_win.get(name, None)
        if win is not None:
            # 已有窗口则追加数据
            self.vis.line( X=x, Y=y, opts=default_opts, update='append',win=win )
        else:
            # 新建窗口
            self.cur_win[name] = self.vis.line( X=x, Y=y, opts=default_opts)

    def vis_image(self, name, img, env=None, opts=None):
        """ 
        可视化图像数据

        参数说明:
            name (str): 图像窗口名称
            img (ndarray或Tensor): 图像数据, 形状为(C, H, W)
            env (str): visdom环境名
            opts (dict): 其他visdom参数
        """
        if env is None:
            env = self.env 
        if self.id is not None:
            name = "[%s]"%self.id + name
        win = self.cur_win.get(name, None)
        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)
        if win is not None:
            # 已有窗口则更新图像
            self.vis.image( img=img, win=win, opts=opts, env=env )
        else:
            # 新建窗口
            self.cur_win[name] = self.vis.image( img=img, opts=default_opts, env=env )
    
    def vis_table(self, name, tbl, opts=None):
        """
        可视化表格数据(如超参数、指标等)

        参数说明:
            name (str): 表格窗口名称
            tbl (dict): 表格内容, key为列名, value为对应值
            opts (dict): 其他visdom参数
        """
        win = self.cur_win.get(name, None)

        # 构造HTML表格字符串
        tbl_str = "<table width=\"100%\"> "
        tbl_str+="<tr> \
                 <th>Term</th> \
                 <th>Value</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str+=  "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>"%(k, v)

        tbl_str+="</table>"

        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)
        if win is not None:
            # 已有窗口则更新表格
            self.vis.text(tbl_str, win=win, opts=default_opts)
        else:
            # 新建窗口
            self.cur_win[name] = self.vis.text(tbl_str, opts=default_opts)


if __name__=='__main__':
    import numpy as np
    vis = Visualizer(port=13500, env='main')
    tbl = {"lr": 214, "momentum": 0.9}
    vis.vis_table("test_table", tbl)
    tbl = {"lr": 244444, "momentum": 0.9, "haha": "hoho"}
    vis.vis_table("test_table", tbl)
