import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham


city_fp = os.path.join(proj_dir, 'data/city.txt')
altitude_fp = os.path.join(proj_dir, 'data/altitude.npy')


class Graph():
    def __init__(self):
        # 距离阈值
        self.dist_thres = 3
        # 海拔阈值
        self.alti_thres = 1200
        self.use_altitude = True

        # 获取海拔原始数据，维度为（641，561），x,y对应一个单元格的海拔高度
        self.altitude = self._load_altitude()
        # 获取节点数据，使用有序字典存储，每个节点包含：城市名、海拔、经度、纬度
        self.nodes = self._gen_nodes()
        # 补充节点的属性值，返回节点海拔的有序nparray
        self.node_attr = self._add_node_attr()
        # 节点数量
        self.node_num = len(self.nodes)
        # 计算不同城市节点之间的测地距离和空间上角度关系
        self.edge_index, self.edge_attr = self._gen_edges()
        # 是否使用海拔高度来重置节点之间关系
        if self.use_altitude:
            self._update_edges()
        # 此处边的数量，包含了两遍边的数值
        self.edge_num = self.edge_index.shape[1]
        # 转化为密集邻接矩阵
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        # 读取altitude文件
        assert os.path.isfile(altitude_fp)
        altitude = np.load(altitude_fp)
        return altitude

    def _lonlat2xy(self, lon, lat, is_aliti):
        """
        将经纬度转化为坐标体系中x，y值，数据转换
        :param lon: 经度
        :param lat: 纬度
        :param is_aliti: 切换起始坐标节点（左上坐标）
        :return:
        """
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05   # 地图分辨率
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125  # 地图分辨率
        # 加上或减去res/2以确保将经纬度转换为平面坐标时，处于格子的中心
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp, 'r') as f:
            for line in f:
                # 从city.txt中读取数据，依次为id，城市名，经度，纬度
                idx, city, lon, lat = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        # 将数据转化为nparray，维度：（184）
        altitude_arr = np.stack(altitude_arr)
        # 变化数据维度，维度：（184，1）
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        # 计算两个集合之间的距离，此处为计算每个城市之间的距离
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        # 对应位置相乘，不是矩阵乘法
        dist = dist * adj
        # 将邻接密集矩阵转化为邻接稀疏矩阵,返回一个稀疏矩阵，和可选批次索引的元组
        # edge_index维度:（2,n)，dist维度：（n），edge_index的每i列代表了dist中第i个元素所处稀疏矩阵中位置
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            # 计算测地距离
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon
            # 将经纬度数转化为m/s
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            # 风速角度计算，实则计算两个城市节点之间的角度偏差
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        # 存在相关性两个城市节点之间的角度关系和距离关系
        attr = np.stack([dist_arr, direc_arr], axis=-1)
        # edge_index代表两个节点之间的开始节点索引和结束节点索引
        return edge_index, attr

    def _update_edges(self):
        # 使用海拔高度更新边属性
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']

            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            # 获取起始点和终结点之间的离散点
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))

            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            # 在城市节点之间存在多个（3个以上）超过1200 阈值的点，即代表两个城市之间没有关联
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:, i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == '__main__':
    graph = Graph()
