class atom:
    def __init__(self):
        self._name=None
        self._num=None
        self._serial_num=None
        self._alt_loc=None
        self._x=None
        self._y=None
        self._z=None
        self._type=None
        self._occupancy=None
        self._bonds=[]
        self._alt_serial_num=[]
        self._stereo_config=None
        self._leaving_atom_flag=None
        self._is_occupied=False
        self._atomic_number=None
        self._properties=None
    @property
    def is_occupied(self):
        return self._is_occupied
    @property
    def properties(self):
        return self._properties
    @property
    def stereo_config(self):
        return self._stereo_config
    @property
    def leaving_atom_flag(self):
        return self._leaving_atom_flag
    @property
    def name(self):
        return self._name
    @property
    def type(self):
        return self._type
    @property
    def serial_num(self):
        return self._serial_num
    @property
    def alt_loc(self):
        return self._alt_loc
    @property
    def coord(self):
        return (self._x,self._y,self._z)
    @property
    def occupancy(self):
        return self._occupancy
    @property
    def bonds(self):
        return self._bonds
    @property
    def alt_serial_num(self):
        return self._alt_serial_num
    @alt_serial_num.setter
    def alt_serial_num(self,value):
        self._alt_serial_num+=[value]
    @properties.setter
    def properties(self,value):
        self._properties=value
    @type.setter
    def type(self,value):
        self._type=value
    @is_occupied.setter
    def is_occupied(self,value):
        self._is_occupied=value
    @name.setter
    def name(self,value):
        self._name=value
    @leaving_atom_flag.setter
    def leaving_atom_flag(self,value):
        self._leaving_atom_flag=value
    @stereo_config.setter
    def stereo_config(self,value):
        self._stereo_config=value
    @serial_num.setter
    def serial_num(self,value):
        self._serial_num=value
    @alt_loc.setter
    def alt_loc(self,value):
        self._alt_loc=value
    @coord.setter
    def coord(self,value):
        self._x,self._y,self._z=value
    @occupancy.setter
    def occupancy(self,value):
        self._occupancy=value
    @bonds.setter
    def bonds(self,value):
        self._bonds+=value
    def is_alpha_carbon(self):
        atms=[str(x) for x in self.bonds if x.stereo_config=='B']
        if len(atms)>0 and str(self)=='C' and self.stereo_config!='N':
            return True
        else:
            return False
    def __str__(self):
        return self.name[1]