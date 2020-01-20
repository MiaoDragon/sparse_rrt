from sparse_rrt import _sst_module
import numpy as np


class WithEuclideanDistanceComputer(object):
    '''
    Add euclidian distance computer to a cpp system class
    '''
    def distance_computer(self):
        return _sst_module.euclidean_distance(np.array(self.is_circular_topology()))


class Car(_sst_module.Car, WithEuclideanDistanceComputer):
    pass


class CartPole(_sst_module.CartPole, WithEuclideanDistanceComputer):
    pass

class RectangleObs(_sst_module.RectangleObsSystem):
    def __init__(self, obstacle_list, obstacle_width, env_name):
        super().__init__(obstacle_list, obstacle_width, env_name)
        self.env_name = env_name
    def distance_computer(self):
        if self.env_name == "acrobot":
            return _sst_module.TwoLinkAcrobotDistance()
        elif self.env_name == 'cartpole':
            return _sst_module.euclidean_distance(np.array(self.is_circular_topology()))

class Pendulum(_sst_module.Pendulum, WithEuclideanDistanceComputer):
    pass


class Point(_sst_module.Point, WithEuclideanDistanceComputer):
    pass


class RallyCar(_sst_module.RallyCar, WithEuclideanDistanceComputer):
    pass


class TwoLinkAcrobot(_sst_module.TwoLinkAcrobot):
    '''
    Acrobot has its own custom distance for faster convergence
    '''
    def distance_computer(self):
        return _sst_module.TwoLinkAcrobotDistance()

class PSOPTCartPole(_sst_module.PSOPTCartPole, WithEuclideanDistanceComputer):
    pass

class PSOPTPendulum(_sst_module.PSOPTPendulum, WithEuclideanDistanceComputer):
    pass
class PSOPTPoint(_sst_module.PSOPTPoint, WithEuclideanDistanceComputer):
    pass
class PSOPTAcrobot(_sst_module.PSOPTAcrobot):
    def distance_computer(self):
        return _sst_module.TwoLinkAcrobotDistance()
